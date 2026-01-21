from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Generic, Optional, TypeVar

from marketdata import MarketData
from schedule import Schedule

StrategyState = TypeVar('StrategyState')

@dataclass(frozen=True)
class Strategy(ABC, Generic[StrategyState]):
    """
    Abstract base class for financial index computation strategies.
    
    This class defines the interface that all index strategies must implement.
    It uses a generic type parameter to allow different strategies to return
    different state types while maintaining type safety.
    
    Type Parameters:
        StrategyState: The type of state object returned by the strategy
    
    Attributes:
        md: MarketData instance providing access to historical price data
    """
    md: MarketData

    @abstractmethod
    def resolve_dates(self, from_date: Optional[date], to_date: date) -> Schedule:
        """
        Resolve a date range into a schedule of valid computation dates.
        
        This method determines which dates within the given range should be
        included in strategy calculations, typically filtering for business days
        or other relevant trading dates.
        
        Args:
            from_date: Start date for the range (None means use strategy's default start)
            to_date: End date for the range (inclusive)
            
        Returns:
            Schedule: A schedule containing the valid dates for index computation
        """
        pass

    @abstractmethod
    def compute_state(self, date: date) -> StrategyState:
        """
        Compute the state for a specific date.
        
        This method calculates all relevant formulae for the index strategy
        on the given date, including the index level, and any
        other strategy-specific state information.
        
        Args:
            date: The date for which to compute the state
            
        Returns:
            StrategyState: Strategy-specific state object containing all computed
                values for the given date
        """
        pass