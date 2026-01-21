from datetime import date
from typing import Dict, Optional
from base import Strategy, StrategyState

def get_states(strategy: Strategy[StrategyState], from_date: Optional[date], to_date: date) -> Dict[date, StrategyState]:
    """
    Get the states for each date in the specified range.
    
    Args:
        strategy: The strategy to compute
        from_date: Start date (None means use strategy's seed date)
        to_date: End date (inclusive)
        
    Returns:
        Dict[date, strategyState]: Dictionary mapping dates to their computed strategy states
    """
    # Resolve the date range using the strategy's calendar
    schedule = strategy.resolve_dates(from_date, to_date)
    
    # Compute strategy state for each date in the schedule
    results = {
        current_date: strategy.compute_state(current_date)
        for current_date in schedule
    }
    
    return results
