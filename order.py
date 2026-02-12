from dataclasses import dataclass
from typing import Optional

@dataclass
class Order:
    stop_loss: Optional[float] = None
    trail_offset: Optional[float] = None
    take_profit: Optional[float] = None

