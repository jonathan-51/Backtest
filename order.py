from dataclasses import dataclass
from typing import Optional

@dataclass
class Order:
    """Exit conditions attached to a trade entry (stop-loss, trailing stop, take-profit)."""
    stop_loss: Optional[float] = None
    trail_offset: Optional[float] = None
    take_profit: Optional[float] = None

