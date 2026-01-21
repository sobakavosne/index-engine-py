"""
StateStore: Generic caching mechanism for strategy states with dependency tracking.
"""
from threading import RLock
from datetime import date
from typing import Dict, Generic, Optional, Set, TypeVar

from base import Strategy
from lock_manager import ThreadingLockManager

StrategyStateType = TypeVar('StrategyStateType')


class StateStore(Generic[StrategyStateType]):
    """
    A generic cache for strategy states that tracks dependencies on market data.
    
    When market data changes, all states that depend on it (directly or indirectly)
    are automatically invalidated.
    
    Thread Safety:
    - All operations are thread-safe when used with a lock manager
    - Cache operations are protected by locks to prevent race conditions
    """
    
    def __init__(self, strategy: Strategy[StrategyStateType], lock_manager: Optional[ThreadingLockManager] = None):
        """
        Initialize StateStore for a strategy.
        
        Args:
            strategy: The strategy instance to cache states for
            lock_manager: Optional lock manager for thread-safe operations.
                        If None, operations are not thread-safe.
        """
        self._strategy = strategy
        self._cache: Dict[date, StrategyStateType] = {}
        # Track which market data each state depends on: {date: {(date, ticker), ...}}
        self._dependencies: Dict[date, Set[tuple[date, str]]] = {}
        # Lock manager for thread-safe operations
        self._lock_manager = lock_manager
        # Internal lock for operations that don't use lock manager
        self._internal_lock = RLock()
    
    def get(self, target_date: date) -> Optional[StrategyStateType]:
        """
        Get a cached state if it exists and is valid.
        
        Thread-safe: Uses lock manager if provided, otherwise uses internal lock.
        
        Args:
            target_date: The date to get the state for
            
        Returns:
            The cached state if valid, None otherwise
        """
        # Use lock manager if available, otherwise use internal lock
        if self._lock_manager:
            with self._lock_manager.acquire_date_lock(target_date):
                return self._get_unsafe(target_date)
        else:
            with self._internal_lock:
                return self._get_unsafe(target_date)
    
    def _get_unsafe(self, target_date: date) -> Optional[StrategyStateType]:
        """
        Internal method to get cached state without locking.
        Must be called with appropriate lock held.
        """
        if target_date in self._cache:
            # Check if all dependencies are still valid
            if self._is_valid(target_date):
                return self._cache[target_date]
            else:
                # Invalidate this state
                del self._cache[target_date]
                if target_date in self._dependencies:
                    del self._dependencies[target_date]
        return None
    
    def put(self, target_date: date, state: StrategyStateType, dependencies: Set[tuple[date, str]]):
        """
        Store a state with its dependencies.
        
        Thread-safe: Uses lock manager if provided, otherwise uses internal lock.
        
        Args:
            target_date: The date this state is for
            state: The computed state
            dependencies: Set of (date, ticker) tuples that this state depends on
        """
        # Use lock manager if available, otherwise use internal lock
        if self._lock_manager:
            with self._lock_manager.acquire_date_lock(target_date):
                self._put_unsafe(target_date, state, dependencies)
        else:
            with self._internal_lock:
                self._put_unsafe(target_date, state, dependencies)
    
    def _put_unsafe(self, target_date: date, state: StrategyStateType, dependencies: Set[tuple[date, str]]):
        """
        Internal method to store state without locking.
        Must be called with appropriate lock held.
        """
        self._cache[target_date] = state
        self._dependencies[target_date] = dependencies.copy()
    
    def invalidate(self, invalidated_date: date):
        """
        Invalidate all states that depend on market data at or after the given date.
        
        Per the spec: when market data at date X changes, all states at date >= X
        must be invalidated because they may depend on it.
        
        Thread-safe: Uses lock manager if provided, otherwise uses internal lock.
        
        Args:
            invalidated_date: The date of market data that changed
        """
        # Use lock manager if available, otherwise use internal lock
        if self._lock_manager:
            with self._lock_manager.acquire_invalidation_lock():
                self._invalidate_unsafe(invalidated_date)
        else:
            with self._internal_lock:
                self._invalidate_unsafe(invalidated_date)
    
    def _invalidate_unsafe(self, invalidated_date: date):
        """
        Internal method to invalidate states without locking.
        Must be called with appropriate lock held.
        """
        # Invalidate all states at this date or later
        dates_to_remove = [
            d for d in self._cache.keys()
            if d >= invalidated_date
        ]
        for d in dates_to_remove:
            del self._cache[d]
            if d in self._dependencies:
                del self._dependencies[d]
    
    def _is_valid(self, target_date: date) -> bool:
        """
        Check if a cached state is still valid.
        
        A state is valid if all market data it depends on is current
        (hasn't been updated since the state was cached).
        
        Note: This method assumes the caller holds the appropriate lock.
        
        Args:
            target_date: The date of the state to check
            
        Returns:
            True if the state is valid, False otherwise
        """
        if target_date not in self._dependencies:
            return False
        
        # Check if any dependency date has been updated in MarketData
        updated_dates = self._strategy.md.get_updated_dates()
        dependency_dates = {d for d, _ in self._dependencies[target_date]}
        
        # If any dependency date has been updated, the state is invalid
        return not (dependency_dates & updated_dates)
    
    def clear(self):
        """
        Clear all cached states.
        
        Thread-safe: Uses internal lock.
        """
        with self._internal_lock:
            self._cache.clear()
            self._dependencies.clear()
