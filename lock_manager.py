"""
Lock Manager: Thread-safe coordination for concurrent access to shared resources.

This module provides lock managers for coordinating concurrent access to shared
resources in the index computation system. The lock manager uses per-date locks
to enable parallel computation of different dates while preventing duplicate
computation of the same date.
"""

from threading import Lock
from contextlib import contextmanager
from datetime import date
from typing import Dict


class ThreadingLockManager:
    """
    Lock manager using Lock for synchronous multi-threaded code.

    This manager provides:
    - Per-date locks for computation (allows parallel computation of different dates)
    - Global invalidation lock (ensures atomic cache invalidation)

    Thread Safety:
    - All operations are thread-safe
    - Per-date locks prevent duplicate computation
    - Invalidation lock prevents race conditions during cache invalidation
    """

    def __init__(self):
        """Initialize the lock manager."""
        # Per-date locks for computation
        self._date_locks: Dict[date, Lock] = {}
        # Lock to protect the date_locks dictionary itself
        self._date_locks_lock = Lock()
        # Global lock for cache invalidation
        self._invalidation_lock = Lock()

    @contextmanager
    def acquire_date_lock(self, target_date: date):
        """
        Acquire a lock for a specific date.

        This context manager ensures that only one thread can compute
        the state for a given date at a time, preventing duplicate computation.

        Args:
            target_date: The date to acquire a lock for

        Yields:
            The lock context (can be used in a 'with' statement)

        Example:
            with lock_manager.acquire_date_lock(date(2023, 1, 5)):
                # Only one thread can execute this block for date 2023-01-05
                state = compute_state(date(2023, 1, 5))
        """
        # Get or create lock for this date
        with self._date_locks_lock:
            if target_date not in self._date_locks:
                self._date_locks[target_date] = Lock()
            date_lock = self._date_locks[target_date]

        # Acquire the date lock
        date_lock.acquire()
        try:
            yield
        finally:
            date_lock.release()

    @contextmanager
    def acquire_invalidation_lock(self):
        """
        Acquire the global invalidation lock.

        This context manager ensures that cache invalidation operations
        are atomic and don't interfere with each other or with computations.

        Yields:
            The lock context (can be used in a 'with' statement)

        Example:
            with lock_manager.acquire_invalidation_lock():
                # Only one thread can invalidate cache at a time
                invalidate_cache(invalidated_date)
        """
        self._invalidation_lock.acquire()
        try:
            yield
        finally:
            self._invalidation_lock.release()
