from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set, Tuple
import datetime

from base import Strategy
from schedule import Schedule
from statestore import StateStore
from lock_manager import ThreadingLockManager

AssetData = Dict[str, float]

@dataclass(frozen=True)
class EqualWeightStrategyState:
    """
    Represents the state of an equal weight strategy at a specific point in time.
    
    Attributes:
        returns: Dictionary mapping asset names to their returns for the period
        portfolio_return: The overall portfolio return for the period
        index_level: The current level/value of the index
        weights: Dictionary mapping asset names to their current portfolio weights
    """
    returns: AssetData
    portfolio_return: float
    index_level: float
    weights: AssetData

@dataclass(frozen=True)
class EqualWeightStrategy(Strategy[EqualWeightStrategyState]):
    """
    An equal weight index strategy that rebalances monthly.
    
    This strategy maintains equal weights across all assets in the basket,
    rebalancing at the end of each month to restore equal weighting.
    
    Attributes:
        basket: List of asset identifiers (tickers) in the strategy
        seed_date: The starting date for index calculation
        calendar: Schedule of valid trading dates
        initial_index_level: Starting value of the index (e.g., 100.0)
    """
    basket: List[str]
    seed_date: date
    calendar: Schedule
    initial_index_level: float
    _state_store: StateStore[EqualWeightStrategyState] = field(init=False, repr=False)
    _lock_manager: Optional[ThreadingLockManager] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize the StateStore for this strategy."""
        object.__setattr__(self, '_state_store', StateStore(self, lock_manager=self._lock_manager))
        # Register callback to invalidate cache when market data is updated
        self.md.register_update_callback(self._state_store.invalidate)
    
    def set_lock_manager(self, lock_manager: ThreadingLockManager):
        """
        Set the lock manager for thread-safe operations.
        
        This should be called after initialization if thread safety is needed.
        The lock manager will be used by the StateStore and should also be
        passed to MarketData if it was created separately.
        
        Args:
            lock_manager: The lock manager instance to use
        """
        object.__setattr__(self, '_lock_manager', lock_manager)
        # Update StateStore with lock manager
        object.__setattr__(self, '_state_store', StateStore(self, lock_manager=lock_manager))
        # Re-register callback
        self.md.register_update_callback(self._state_store.invalidate)

    def resolve_dates(self, from_date: Optional[date], to_date: date) -> Schedule:
        """
        Get a schedule of dates within the specified range.
        
        Args:
            from_date: Start date (defaults to seed_date if None)
            to_date: End date (inclusive)
            
        Returns:
            Schedule: Sub-schedule containing dates in the specified range
        """
        if from_date is None:
            from_date = self.seed_date
        
        return self.calendar.sub_schedule(from_date, to_date)

    def compute_state(self, date: date) -> EqualWeightStrategyState:
        """
        Compute the index state for a given date.
        
        This method incrementally calculates the index state by:
        1. Starting from the seed date with initial conditions
        2. Computing daily returns for each asset
        3. Calculating portfolio return using previous day's weights
        4. Updating index level based on portfolio return
        5. Rebalancing weights to equal weight at month-end
        
        Thread Safety:
        - If a lock manager is set, this method uses per-date locks to prevent
          duplicate computation of the same date by multiple threads.
        - The lock is acquired before checking the cache and released after
          storing the result, ensuring atomicity.
        
        Args:
            date: The date for which to compute the index state
            
        Returns:
            EqualWeightStrategyState: The complete state of the strategy on the given date
        """
        # Use lock manager if available to prevent duplicate computation
        if self._lock_manager:
            with self._lock_manager.acquire_date_lock(date):
                return self._compute_state_unsafe(date)
        else:
            return self._compute_state_unsafe(date)
    
    def _compute_state_unsafe(self, date: date) -> EqualWeightStrategyState:
        """
        Internal method to compute state without locking.
        Must be called with appropriate lock held if thread safety is needed.
        """
        # Check StateStore cache first
        # Use unsafe version if lock manager is set (we're already holding the lock)
        # Otherwise use safe version (which uses internal lock)
        if self._lock_manager:
            cached_state = self._state_store._get_unsafe(date)
        else:
            cached_state = self._state_store.get(date)
        if cached_state is not None:
            return cached_state
        
        # Declare and initialize dependencies once at function scope
        dependencies: Set[Tuple[datetime.date, str]] = set()
        
        if date == self.seed_date:
            # Base case: return initial state at seed date
            state = EqualWeightStrategyState(
                returns={asset: 0.0 for asset in self.basket},
                portfolio_return=0.0,
                index_level=self.initial_index_level,
                weights={asset: 1/len(self.basket) for asset in self.basket},
            )
            # Seed date doesn't depend on market data (dependencies already empty)
            # Use unsafe version if lock manager is set (we're already holding the lock)
            if self._lock_manager:
                self._state_store._put_unsafe(date, state, dependencies)
            else:
                self._state_store.put(date, state, dependencies)
            return state

        # Incremental case: compute based on previous day
        prev_date = self.calendar.prev(date)
        # Recursive call - use compute_state() to ensure proper locking for prev_date
        # This is safe because we're acquiring a different lock (for prev_date) while
        # holding the lock for date - no deadlock since they're different locks
        if self._lock_manager:
            # Use compute_state to get proper locking for prev_date
            prev_state = self.compute_state(prev_date)
        else:
            # No lock manager, use unsafe version
            prev_state = self._compute_state_unsafe(prev_date)

        # Calculate daily returns for each asset: (today_price / yesterday_price) - 1
        # Track dependencies: state at date depends on market data at date and prev_date
        returns: AssetData = {}
        # Clear and repopulate dependencies for incremental case
        dependencies.clear()
        for asset in self.basket:
            returns[asset] = self.md.get(date, asset) / self.md.get(prev_date, asset) - 1
            # Track dependencies on market data
            dependencies.add((date, asset))
            dependencies.add((prev_date, asset))

        # Calculate portfolio return as weighted sum of asset returns
        portfolio_return = sum([returns[asset] * weight for asset, weight in prev_state.weights.items()])
        index_level = prev_state.index_level * (1 + portfolio_return)

        # Rebalance weights at end of month, otherwise let them drift
        if self.calendar.is_last_day_of_month(date):
            # Rebalance to equal weights (1/n for each asset)
            weights = {asset: 1/len(self.basket) for asset in self.basket}
        else:
            # Recalculate weights based on price movements
            # Each weight is adjusted by the return of that asset, normalized to sum to 1
            weights = {
                asset: prev_state.weights[asset] * (1 + returns[asset]) / (1 + portfolio_return)
                for asset in self.basket
            }

        # Return the calculated state
        state = EqualWeightStrategyState(
            returns=returns,
            portfolio_return=portfolio_return,
            index_level=index_level,
            weights=weights,
        )
        # Store in StateStore with dependencies
        # Use unsafe version if lock manager is set (we're already holding the lock)
        if self._lock_manager:
            self._state_store._put_unsafe(date, state, dependencies)
        else:
            self._state_store.put(date, state, dependencies)
        return state
