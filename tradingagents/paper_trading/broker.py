"""Abstract Broker interface — implementations: alpaca_broker, simulator_broker.

The TradingAgents pipeline consumes a single ``Broker`` to fetch account state
and submit orders. The interface is deliberately minimal so a swap between
Alpaca paper, our local simulator, or a future IBKR adapter is one line of
config rather than a refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float


@dataclass(frozen=True)
class Account:
    cash: float
    equity: float          # cash + market value of positions
    buying_power: float    # often 2x equity for margin paper accounts


@dataclass(frozen=True)
class OrderResult:
    symbol: str
    qty: float
    side: Literal["buy", "sell"]
    status: str            # "filled" | "accepted" | "rejected" | etc.
    filled_avg_price: float | None
    broker_order_id: str | None


class Broker(Protocol):
    """Minimal broker surface — fetches state, prices, submits market orders."""

    name: str

    def get_account(self) -> Account: ...

    def get_positions(self) -> list[Position]: ...

    def get_price(self, symbol: str) -> float:
        """Last trade / close price used for sizing decisions before an order goes in."""
        ...

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
    ) -> OrderResult:
        """Submit a market-day order. Implementations may translate to MOO/MOC if
        the market is closed and the broker supports it; otherwise they raise."""
        ...
