"""TradingAgents-decision → broker order mapping.

Strategy: target a fixed equity allocation per ticker based on the 5-tier
rating. Compute the delta from the current position and submit a market
order for the difference. Conservative defaults — Buy only goes up to 20%
of total equity per ticker, so a 5-ticker watchlist with all Buys still
leaves dry powder.

Tier targets (% of total equity):
    Buy           20%
    Overweight    12%
    Hold          (no change to existing position)
    Underweight    5%
    Sell           0%   (close)

Override via `TierTargets(...)` if you want a different sizing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from .broker import Broker, OrderResult

logger = logging.getLogger(__name__)

Rating = Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]


@dataclass(frozen=True)
class TierTargets:
    buy: float = 0.20
    overweight: float = 0.12
    underweight: float = 0.05
    sell: float = 0.0
    # Hold deliberately has no entry — we leave the position untouched.

    def target_for(self, rating: Rating) -> float | None:
        return {
            "Buy": self.buy,
            "Overweight": self.overweight,
            "Underweight": self.underweight,
            "Sell": self.sell,
        }.get(rating)


def _normalize_rating(decision: str) -> Rating | None:
    """Pull a tier rating out of a TradingAgents decision string.

    The graph's signal processor returns rendered markdown that begins with
    something like ``**Rating**: Overweight``. Be defensive — accept the bare
    word too in case a caller pre-processed it.
    """
    if not isinstance(decision, str):
        return None
    text = decision.strip()
    # Look for a "Rating: Foo" line first.
    for line in text.splitlines():
        cleaned = line.strip().lstrip("*").lstrip()
        if cleaned.lower().startswith("rating"):
            after = cleaned.split(":", 1)[1].strip().strip("*").strip() if ":" in cleaned else ""
            if after:
                text = after
                break
    head = text.split()[0] if text.split() else ""
    head_lc = head.lower()
    for canonical in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
        if head_lc == canonical.lower():
            return canonical  # type: ignore[return-value]
    return None


def execute_decision(
    broker: Broker,
    symbol: str,
    decision: str,
    *,
    targets: TierTargets | None = None,
    fractional: bool = True,
) -> OrderResult | None:
    """Translate a TradingAgents decision into a broker order.

    Returns the resulting OrderResult, or None if Hold (or unparseable).
    """
    rating = _normalize_rating(decision)
    if rating is None:
        logger.warning("execute_decision(%s): no rating found in decision; skipping.", symbol)
        return None
    if rating == "Hold":
        logger.info("execute_decision(%s): Hold — no order.", symbol)
        return None

    targets = targets or TierTargets()
    target_pct = targets.target_for(rating)
    if target_pct is None:
        return None

    account = broker.get_account()
    target_value = account.equity * target_pct

    # Current holding (0 if not held).
    current_qty = 0.0
    for pos in broker.get_positions():
        if pos.symbol.upper() == symbol.upper():
            current_qty = pos.qty
            break

    price = broker.get_price(symbol)
    if price <= 0:
        logger.warning("execute_decision(%s): non-positive price %s; skipping.", symbol, price)
        return None

    target_qty = target_value / price
    delta_qty = target_qty - current_qty

    # Round whole shares unless the broker supports fractionals; tiny deltas
    # below one share aren't worth a market order.
    min_delta = 0.01 if fractional else 1.0
    if abs(delta_qty) < min_delta:
        logger.info(
            "execute_decision(%s): delta %.4f below minimum %.2f; no order.",
            symbol, delta_qty, min_delta,
        )
        return None

    qty = round(abs(delta_qty), 4 if fractional else 0)
    if not fractional:
        qty = int(qty)
        if qty == 0:
            return None

    side: Literal["buy", "sell"] = "buy" if delta_qty > 0 else "sell"
    logger.info(
        "execute_decision(%s): rating=%s target=%.2f%% qty=%.4f side=%s price=%.2f",
        symbol, rating, target_pct * 100, qty, side, price,
    )
    return broker.submit_market_order(symbol, qty, side)
