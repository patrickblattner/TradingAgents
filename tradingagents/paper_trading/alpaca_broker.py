"""Alpaca paper-trading broker (alpaca-py SDK).

Reads `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, and `APCA_API_BASE_URL` from
the environment. The base URL must point at the paper endpoint
(https://paper-api.alpaca.markets/v2) — pointing this at the live URL while
keys are paper-keyed (or vice versa) returns a confusing 403 rather than a
clear error.
"""

from __future__ import annotations

import os
from typing import Literal

from .broker import Account, Broker, OrderResult, Position

PAPER_BASE_URL = "https://paper-api.alpaca.markets/v2"


class AlpacaBrokerError(RuntimeError):
    """Raised when Alpaca credentials are missing or the SDK can't be imported."""


def _require_credentials() -> tuple[str, str, str]:
    key = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    base = os.environ.get("APCA_API_BASE_URL", PAPER_BASE_URL).strip()
    if not key or not secret:
        raise AlpacaBrokerError(
            "Alpaca credentials missing. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "in your .env (paper trading endpoint expected). Use --broker simulator "
            "to fall back to the local file-based simulator."
        )
    return key, secret, base


class AlpacaBroker:
    """Paper-trading wrapper around alpaca-py's TradingClient."""

    name = "alpaca"

    def __init__(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise AlpacaBrokerError(
                "alpaca-py is not installed. Run `uv add alpaca-py` (or `pip install alpaca-py`)."
            ) from exc

        key, secret, base = _require_credentials()
        # paper=True wires the SDK to the paper endpoint and disables a few
        # live-only behaviours; we still pass the explicit URL so a custom
        # endpoint (sandbox, proxy) takes precedence over the SDK default.
        self._client = TradingClient(api_key=key, secret_key=secret, paper=True, url_override=base)

    def get_account(self) -> Account:
        acc = self._client.get_account()
        return Account(
            cash=float(acc.cash),
            equity=float(acc.equity),
            buying_power=float(acc.buying_power),
        )

    def get_positions(self) -> list[Position]:
        out: list[Position] = []
        for p in self._client.get_all_positions():
            out.append(Position(
                symbol=str(p.symbol),
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
            ))
        return out

    def get_price(self, symbol: str) -> float:
        # Alpaca's paper TradingClient doesn't expose quotes directly — use the
        # market data SDK if available, else fall back to yfinance for a price
        # close enough for sizing decisions. The actual fill happens at the
        # broker, so a small drift here is fine.
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest

            key, secret, _ = _require_credentials()
            data_client = StockHistoricalDataClient(api_key=key, secret_key=secret)
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            latest = data_client.get_stock_latest_trade(req)
            return float(latest[symbol].price)
        except Exception:
            import yfinance as yf
            return float(yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1])

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
    ) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        filled_price = (
            float(order.filled_avg_price)
            if getattr(order, "filled_avg_price", None) is not None
            else None
        )
        return OrderResult(
            symbol=symbol,
            qty=qty,
            side=side,
            status=str(order.status),
            filled_avg_price=filled_price,
            broker_order_id=str(order.id) if getattr(order, "id", None) else None,
        )
