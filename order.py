from dataclasses import dataclass
from typing import Optional

@dataclass
class Order:
    """Exit conditions attached to a trade entry (stop-loss, take-profit)"""
    stop_loss:Optional[float] = None
    take_profit:Optional[float] = None

@dataclass
class Position:
    """Position details"""
    symbol:str
    direction:str
    entry_date:str
    entry_price:str
    shares:int
    commission_in:float
    stop_loss:Optional[float] = None
    take_profit:Optional[float] = None