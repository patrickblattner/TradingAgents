"""Local JSON-file paper-trading simulator.

Holds a portfolio in ``~/.tradingagents/paper_portfolio.json`` and uses
yfinance for last-trade prices. This is the fallback that works for any user
in any country without a broker account, and is also useful for backtesting
where deterministic, free fills matter more than realism.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import yfinance as yf

from .broker import Account, Broker, OrderResult, Position

DEFAULT_PORTFOLIO_PATH = Path.home() / ".tradingagents" / "paper_portfolio.json"
DEFAULT_STARTING_CASH = 100_000.0


def _portfolio_path() -> Path:
    override = os.environ.get("TRADINGAGENTS_PAPER_PORTFOLIO_PATH")
    return Path(override).expanduser() if override else DEFAULT_PORTFOLIO_PATH


def _empty_portfolio(starting_cash: float) -> dict:
    return {
        "cash": starting_cash,
        "positions": {},   # symbol -> {qty, avg_entry_price}
        "history": [],     # list of fills
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _load(starting_cash: float) -> dict:
    path = _portfolio_path()
    if not path.is_file():
        return _empty_portfolio(starting_cash)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_portfolio(starting_cash)


def _save(data: dict) -> None:
    path = _portfolio_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _last_price(symbol: str) -> float:
    """Latest trade price via yfinance. Falls back to last 1-day close."""
    ticker = yf.Ticker(symbol)
    fast = getattr(ticker, "fast_info", None)
    if fast:
        price = fast.get("last_price") if hasattr(fast, "get") else getattr(fast, "last_price", None)
        if price:
            return float(price)
    hist = ticker.history(period="1d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No price available for {symbol}")
    return float(hist["Close"].iloc[-1])


class SimulatorBroker:
    """JSON-file portfolio + yfinance prices, no broker account required."""

    name = "simulator"

    def __init__(self, *, starting_cash: float = DEFAULT_STARTING_CASH) -> None:
        self._starting_cash = starting_cash
        self._state = _load(starting_cash)

    def _market_value(self) -> float:
        total = 0.0
        for symbol, holding in self._state.get("positions", {}).items():
            qty = float(holding.get("qty", 0.0))
            if qty == 0:
                continue
            try:
                price = _last_price(symbol)
            except Exception:
                # Stale entry — skip rather than fail the whole account fetch.
                continue
            total += qty * price
        return total

    def get_account(self) -> Account:
        cash = float(self._state.get("cash", 0.0))
        market_value = self._market_value()
        equity = cash + market_value
        # Simulator: no margin, buying power = cash. Documented intentionally.
        return Account(cash=cash, equity=equity, buying_power=cash)

    def get_positions(self) -> list[Position]:
        out: list[Position] = []
        for symbol, holding in self._state.get("positions", {}).items():
            qty = float(holding.get("qty", 0.0))
            if qty == 0:
                continue
            try:
                price = _last_price(symbol)
            except Exception:
                price = float(holding.get("avg_entry_price", 0.0))
            out.append(Position(
                symbol=symbol,
                qty=qty,
                avg_entry_price=float(holding.get("avg_entry_price", 0.0)),
                market_value=qty * price,
            ))
        return out

    def get_price(self, symbol: str) -> float:
        return _last_price(symbol)

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
    ) -> OrderResult:
        if qty <= 0:
            raise ValueError("qty must be positive; use side to indicate direction")
        price = _last_price(symbol)
        positions = self._state.setdefault("positions", {})
        holding = positions.setdefault(symbol, {"qty": 0.0, "avg_entry_price": 0.0})
        cur_qty = float(holding["qty"])
        cur_avg = float(holding["avg_entry_price"])

        if side == "buy":
            cost = qty * price
            if cost > self._state["cash"]:
                return OrderResult(
                    symbol=symbol, qty=qty, side=side, status="rejected",
                    filled_avg_price=None, broker_order_id=None,
                )
            new_qty = cur_qty + qty
            new_avg = (cur_qty * cur_avg + qty * price) / new_qty if new_qty else 0.0
            holding["qty"] = new_qty
            holding["avg_entry_price"] = new_avg
            self._state["cash"] -= cost
        else:  # sell
            if qty > cur_qty:
                return OrderResult(
                    symbol=symbol, qty=qty, side=side, status="rejected",
                    filled_avg_price=None, broker_order_id=None,
                )
            holding["qty"] = cur_qty - qty
            self._state["cash"] += qty * price
            if holding["qty"] == 0:
                holding["avg_entry_price"] = 0.0

        order_id = f"sim_{uuid.uuid4().hex[:12]}"
        fill = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "symbol": symbol, "qty": qty, "side": side,
            "price": price, "order_id": order_id,
        }
        self._state.setdefault("history", []).append(fill)
        _save(self._state)
        return OrderResult(
            symbol=symbol, qty=qty, side=side, status="filled",
            filled_avg_price=price, broker_order_id=order_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Snapshot for reporting/debugging — does not persist."""
        return {
            "cash": self._state["cash"],
            "positions": dict(self._state.get("positions", {})),
            "history_count": len(self._state.get("history", [])),
        }
