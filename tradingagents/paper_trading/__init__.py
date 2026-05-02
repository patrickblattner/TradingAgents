"""Paper-trading layer for TradingAgents.

Two backends, swappable via config or CLI flag:

- ``alpaca_broker.AlpacaBroker``  — Alpaca paper-trading account (real broker fills)
- ``simulator_broker.SimulatorBroker`` — local JSON portfolio + yfinance prices
  (works in any country, no broker account)

The ``executor.execute_decision`` function maps a TradingAgents 5-tier rating
to a target equity allocation and submits the delta order.
"""

from __future__ import annotations

from .broker import Account, Broker, OrderResult, Position
from .executor import TierTargets, execute_decision
from .simulator_broker import SimulatorBroker

__all__ = [
    "Account",
    "Broker",
    "OrderResult",
    "Position",
    "SimulatorBroker",
    "TierTargets",
    "execute_decision",
    "build_broker",
]


def build_broker(name: str = "simulator") -> Broker:
    """Resolve a broker by name. Defaults to simulator (works without setup)."""
    name_lc = name.lower()
    if name_lc == "simulator":
        return SimulatorBroker()
    if name_lc == "alpaca":
        from .alpaca_broker import AlpacaBroker
        return AlpacaBroker()
    raise ValueError(f"Unknown broker: {name}. Choose 'simulator' or 'alpaca'.")
