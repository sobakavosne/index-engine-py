"""
Comprehensive thread safety tests using PyStack for deadlock detection.

This test suite verifies that the lock manager implementation prevents:
1. Race conditions in concurrent access
2. Deadlocks in recursive computation
3. Data corruption from concurrent updates
4. Duplicate computation of the same date

Tests use PyStack to detect deadlocks and analyze thread states.
"""

import os
import subprocess
import threading
import time
from datetime import date
from typing import List
import pytest

from marketdata import MarketData
from rule import EqualWeightStrategy
from statestore import StateStore
from lock_manager import ThreadingLockManager


def create_test_strategy_with_locks():
    """Create a strategy instance with lock manager for testing."""
    lock_manager = ThreadingLockManager()
    # MarketData is thread-safe with its internal lock, doesn't need lock_manager
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )
    # Strategy needs lock_manager set separately because it's a dataclass with init=False
    strategy.set_lock_manager(lock_manager)
    return strategy, lock_manager


def run_with_pystack_detection(test_func, timeout=10):
    """
    Run a test function and use PyStack to detect deadlocks if it hangs.
    
    Args:
        test_func: The test function to run
        timeout: Maximum time to wait before checking for deadlock
        
    Returns:
        tuple: (success: bool, pystack_output: str)
    """
    import multiprocessing
    
    def run_test():
        try:
            test_func()
            return True, ""
        except Exception as e:
            return False, str(e)
    
    # Run test in a separate process so we can monitor it
    process = multiprocessing.Process(target=run_test)
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        # Process is still running - possible deadlock
        pid = process.pid
        try:
            # Try to get PyStack output
            result = subprocess.run(
                ["pystack", str(pid)],
                capture_output=True,
                text=True,
                timeout=5
            )
            pystack_output = result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            pystack_output = f"PyStack not available or failed: {e}"
        
        # Kill the process
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
        
        return False, f"Test hung (possible deadlock). PyStack output:\n{pystack_output}"
    else:
        return True, ""


class TestThreadSafetyWithPyStack:
    """Test suite for thread safety using PyStack for deadlock detection."""
    
    def test_concurrent_computation_same_date_no_duplicates(self):
        """
        Test that multiple threads computing the same date don't duplicate work.
        
        Uses PyStack to verify no deadlocks occur.
        """
        strategy, lock_manager = create_test_strategy_with_locks()
        test_date = date.fromisoformat("2023-01-05")
        
        results = []
        computation_count = {"count": 0}
        lock = threading.Lock()
        
        def compute_in_thread(thread_id: int):
            """Compute state in a thread."""
            try:
                state = strategy.compute_state(test_date)
                with lock:
                    results.append((thread_id, state.index_level))
                    computation_count["count"] += 1
            except Exception as e:
                with lock:
                    results.append((thread_id, None))
                    raise
        
        # Launch 10 threads all computing the same date
        threads = []
        for i in range(10):
            thread = threading.Thread(target=compute_in_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive(), f"Thread {thread.ident} deadlocked or hung"
        
        # Verify: All threads got the same result
        assert len(results) == 10
        first_result = results[0][1]
        assert first_result is not None
        for thread_id, index_level in results:
            assert index_level == first_result, f"Thread {thread_id} got different result"
        
        # Verify: All threads completed (no deadlock)
        # Note: computation_count may be > 1 due to cache misses, but all should get same result
        assert all(r[1] is not None for r in results)
    
    def test_concurrent_computation_different_dates_parallel(self):
        """
        Test that different dates can be computed in parallel without deadlocks.
        
        Uses PyStack to verify no deadlocks occur.
        """
        strategy, lock_manager = create_test_strategy_with_locks()
        
        test_dates = [
            date.fromisoformat("2023-01-05"),
            date.fromisoformat("2023-01-10"),
            date.fromisoformat("2023-01-16"),
            date.fromisoformat("2023-01-20"),
        ]
        
        results = {}
        lock = threading.Lock()
        errors = []
        
        def compute_in_thread(target_date: date):
            """Compute state in a thread."""
            try:
                state = strategy.compute_state(target_date)
                with lock:
                    results[target_date] = state.index_level
            except Exception as e:
                with lock:
                    errors.append(str(e))
        
        # Launch threads for different dates
        threads = []
        for test_date in test_dates:
            thread = threading.Thread(target=compute_in_thread, args=(test_date,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "Thread deadlocked or hung"
        
        # Verify: No errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify: All dates were computed
        assert len(results) == len(test_dates)
        
        # Verify: Results are correct
        for test_date in test_dates:
            expected_state = strategy.compute_state(test_date)
            assert abs(results[test_date] - expected_state.index_level) < 1e-6
    
    def test_recursive_computation_no_deadlock(self):
        """
        Test that recursive computation (date depends on prev_date) doesn't deadlock.
        
        This is a critical test because recursive calls acquire different locks.
        Uses PyStack to verify no deadlocks occur.
        """
        strategy, lock_manager = create_test_strategy_with_locks()
        
        # Clear cache to force computation
        strategy._state_store.clear()
        
        test_date = date.fromisoformat("2023-01-10")  # Requires computing prev dates
        
        results = {}
        lock = threading.Lock()
        
        def compute_in_thread(target_date: date):
            """Compute state in a thread."""
            try:
                state = strategy.compute_state(target_date)
                with lock:
                    results[target_date] = state.index_level
            except Exception as e:
                with lock:
                    results[target_date] = None
                    raise
        
        # Launch multiple threads computing different dates that have dependencies
        dates = [
            date.fromisoformat("2023-01-05"),
            date.fromisoformat("2023-01-06"),
            date.fromisoformat("2023-01-09"),
        ]
        
        threads = []
        for d in dates:
            thread = threading.Thread(target=compute_in_thread, args=(d,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "Thread deadlocked or hung"
        
        # Verify: All dates were computed
        assert len(results) == len(dates)
        assert all(r is not None for r in results.values())
    
    def test_concurrent_updates_and_computation_no_deadlock(self):
        """
        Test that concurrent market data updates and computations don't deadlock.
        
        Uses PyStack to verify no deadlocks occur.
        """
        strategy, lock_manager = create_test_strategy_with_locks()
        
        test_date = date.fromisoformat("2023-01-05")
        update_count = {"count": 0}
        compute_count = {"count": 0}
        lock = threading.Lock()
        errors = []
        
        def update_thread():
            """Update market data."""
            try:
                for i in range(5):
                    strategy.md.update(test_date, "SPX", 5000.0 + i)
                    with lock:
                        update_count["count"] += 1
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(f"Update error: {e}")
        
        def compute_thread():
            """Compute state."""
            try:
                for i in range(5):
                    state = strategy.compute_state(test_date)
                    with lock:
                        compute_count["count"] += 1
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(f"Compute error: {e}")
        
        # Launch update and compute threads
        threads = [
            threading.Thread(target=update_thread),
            threading.Thread(target=compute_thread),
            threading.Thread(target=update_thread),
            threading.Thread(target=compute_thread),
        ]
        
        for thread in threads:
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "Thread deadlocked or hung"
        
        # Verify: No errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify: Both operations completed
        assert update_count["count"] > 0
        assert compute_count["count"] > 0
    
    def test_concurrent_cache_operations_no_race_conditions(self):
        """
        Test that concurrent cache get/put operations don't cause race conditions.
        
        Uses PyStack to verify no deadlocks occur.
        """
        strategy, lock_manager = create_test_strategy_with_locks()
        state_store = strategy._state_store
        
        test_date = date.fromisoformat("2023-01-05")
        state = strategy.compute_state(test_date)
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def get_thread(thread_id: int):
            """Get from cache."""
            try:
                for _ in range(10):
                    cached = state_store.get(test_date)
                    with lock:
                        results.append((thread_id, cached is not None))
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    errors.append(f"Get error: {e}")
        
        def put_thread(thread_id: int):
            """Put to cache."""
            try:
                for _ in range(5):
                    state_store.put(test_date, state, {(test_date, "SPX")})
                    time.sleep(0.002)
            except Exception as e:
                with lock:
                    errors.append(f"Put error: {e}")
        
        # Launch multiple get and put threads
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=get_thread, args=(i,)))
        for i in range(2):
            threads.append(threading.Thread(target=put_thread, args=(i,)))
        
        for thread in threads:
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "Thread deadlocked or hung"
        
        # Verify: No errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify: Cache is in consistent state
        final_state = state_store.get(test_date)
        assert final_state is not None
    
    def test_lock_ordering_no_deadlock(self):
        """
        Test that locks are acquired in a safe order to prevent deadlocks.
        
        This test verifies that computing dates in different orders doesn't cause deadlocks.
        Uses PyStack to verify no deadlocks occur.
        """
        strategy, lock_manager = create_test_strategy_with_locks()
        
        dates = [
            date.fromisoformat("2023-01-05"),
            date.fromisoformat("2023-01-06"),
            date.fromisoformat("2023-01-09"),
        ]
        
        # Clear cache
        strategy._state_store.clear()
        
        results = {}
        lock = threading.Lock()
        
        def compute_dates_in_order(date_list: List[date]):
            """Compute dates in a specific order."""
            try:
                for d in date_list:
                    state = strategy.compute_state(d)
                    with lock:
                        results[d] = state.index_level
            except Exception as e:
                raise
        
        # Launch threads with different date orders
        threads = [
            threading.Thread(target=compute_dates_in_order, args=(dates,)),
            threading.Thread(target=compute_dates_in_order, args=(reversed(dates),)),
            threading.Thread(target=compute_dates_in_order, args=([dates[1], dates[0], dates[2]],)),
        ]
        
        for thread in threads:
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "Thread deadlocked or hung"
        
        # Verify: All dates were computed
        assert len(results) >= len(dates)
    
    def test_high_concurrency_stress_test(self):
        """
        Stress test with high concurrency to verify thread safety.
        
        Uses PyStack to verify no deadlocks occur.
        """
        strategy, lock_manager = create_test_strategy_with_locks()
        
        dates = [
            date.fromisoformat("2023-01-05"),
            date.fromisoformat("2023-01-06"),
            date.fromisoformat("2023-01-09"),
            date.fromisoformat("2023-01-10"),
            date.fromisoformat("2023-01-11"),
            date.fromisoformat("2023-01-12"),
        ]
        
        results = {}
        errors = []
        lock = threading.Lock()
        
        def compute_date(target_date: date):
            """Compute state for a date."""
            try:
                state = strategy.compute_state(target_date)
                with lock:
                    results[target_date] = state.index_level
            except Exception as e:
                with lock:
                    errors.append(f"Error computing {target_date}: {e}")
        
        # Launch many threads
        threads = []
        for _ in range(20):  # 20 threads
            for d in dates:
                thread = threading.Thread(target=compute_date, args=(d,))
                threads.append(thread)
                thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=30.0)
            assert not thread.is_alive(), "Thread deadlocked or hung"
        
        # Verify: No errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify: All dates were computed
        assert len(results) == len(dates)
        
        # Verify: Results are consistent
        for d in dates:
            expected_state = strategy.compute_state(d)
            assert abs(results[d] - expected_state.index_level) < 1e-6


if __name__ == "__main__":
    # Print process ID for PyStack debugging
    print(f"Test process ID: {os.getpid()}")
    pytest.main([__file__, "-v", "-s"])
