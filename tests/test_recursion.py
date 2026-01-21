"""
Tests for recursion behavior and stack overflow prevention in compute_state.
"""
import os
import sys
import tempfile
from datetime import date

import pytest
from marketdata import MarketData
from rule import EqualWeightStrategy


def create_strategy():
    """Create a strategy instance for testing."""
    md = MarketData("sample_prices.csv")
    return EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )


def test_recursion_terminates_at_seed_date():
    """Test that recursion correctly terminates at seed_date."""
    strategy = create_strategy()

    target_date = date.fromisoformat("2023-06-29")

    state = strategy.compute_state(target_date)

    assert state is not None
    assert state.index_level > 0
    assert len(state.weights) == len(strategy.basket)


def test_deep_recursion_without_cache():
    """Test that deep recursion works even without cache (worst case)."""
    strategy = create_strategy()

    strategy._state_store.clear()  # type: ignore

    target_date = date.fromisoformat("2023-06-29")

    calendar = strategy.calendar
    dates_list = list(calendar)
    seed_idx = dates_list.index(strategy.seed_date)
    target_idx = dates_list.index(target_date)
    depth = target_idx - seed_idx

    assert depth < sys.getrecursionlimit(), f"Recursion depth {depth} exceeds limit {sys.getrecursionlimit()}"

    state = strategy.compute_state(target_date)
    assert state is not None


def test_cache_reduces_recursion_depth():
    """Test that caching reduces effective recursion depth."""
    strategy = create_strategy()

    dates = [
        date.fromisoformat("2023-01-03"),
        date.fromisoformat("2023-01-04"),
        date.fromisoformat("2023-01-05"),
        date.fromisoformat("2023-01-06"),
    ]

    for d in dates:
        strategy.compute_state(d)

    later_date = date.fromisoformat("2023-01-10")
    state = strategy.compute_state(later_date)

    assert state is not None
    assert strategy._state_store.get(dates[0]) is not None  # type: ignore


def test_recursion_with_very_long_range():
    """Test recursion with a very long range in the dataset."""
    strategy = create_strategy()

    calendar = strategy.calendar
    dates_list = list(calendar)
    last_date = dates_list[-2] if len(dates_list) > 1 else dates_list[-1]

    seed_idx = dates_list.index(strategy.seed_date)
    last_idx = dates_list.index(last_date)
    depth = last_idx - seed_idx

    assert depth < sys.getrecursionlimit()

    state = strategy.compute_state(last_date)
    assert state is not None
    assert state.index_level > 0


def test_recursion_base_case():
    """Test that recursion base case (seed_date) works correctly."""
    strategy = create_strategy()

    strategy._state_store.clear()  # type: ignore

    state = strategy.compute_state(strategy.seed_date)

    assert state is not None
    assert state.index_level == strategy.initial_index_level
    assert state.portfolio_return == 0.0
    assert all(ret == 0.0 for ret in state.returns.values())


def test_recursion_one_day_after_seed():
    """Test recursion for date immediately after seed_date."""
    strategy = create_strategy()

    strategy._state_store.clear()  # type: ignore

    seed_idx = list(strategy.calendar).index(strategy.seed_date)
    next_date = list(strategy.calendar)[seed_idx + 1]

    state = strategy.compute_state(next_date)

    assert state is not None
    assert strategy._state_store.get(strategy.seed_date) is not None  # type: ignore
    assert strategy._state_store.get(next_date) is not None  # type: ignore


def test_recursion_prevents_infinite_loop():
    """Test that recursion doesn't cause infinite loop."""
    strategy = create_strategy()

    strategy._state_store.clear()  # type: ignore

    target_date = date.fromisoformat("2023-01-10")

    import time

    start = time.perf_counter()
    state = strategy.compute_state(target_date)
    elapsed = time.perf_counter() - start

    assert state is not None
    assert elapsed < 1.0, f"Computation took {elapsed:.2f}s, possible infinite loop"


def test_recursion_with_cache_invalidation():
    """Test recursion behavior when cache is invalidated during computation."""
    strategy = create_strategy()

    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")
    date3 = date.fromisoformat("2023-01-05")

    _ = strategy.compute_state(date1)
    _ = strategy.compute_state(date2)
    state3 = strategy.compute_state(date3)

    strategy.md.update(date1, "SPX", strategy.md.get(date1, "SPX") * 1.1)

    state3_new = strategy.compute_state(date3)

    assert state3_new is not None
    assert state3_new.index_level != state3.index_level  # Should be recomputed


def test_recursion_depth_measurement():
    """Test to measure actual recursion depth."""
    strategy = create_strategy()

    strategy._state_store.clear()  # type: ignore

    test_dates = [
        date.fromisoformat("2023-01-03"),  # 1 day after seed
        date.fromisoformat("2023-01-10"),  # ~8 days after seed
        date.fromisoformat("2023-02-01"),  # ~1 month after seed
        date.fromisoformat("2023-06-29"),  # ~6 months after seed (not last date)
    ]

    calendar = strategy.calendar
    dates_list = list(calendar)
    seed_idx = dates_list.index(strategy.seed_date)

    for target_date in test_dates:
        if target_date in dates_list:
            target_idx = dates_list.index(target_date)
            expected_depth = target_idx - seed_idx

            assert expected_depth < sys.getrecursionlimit(), (
                f"Date {target_date} requires depth {expected_depth}, "
                f"exceeds limit {sys.getrecursionlimit()}"
            )

            state = strategy.compute_state(target_date)
            assert state is not None


def test_recursion_with_sequential_computation():
    """Test that sequential computation (which uses cache) prevents deep recursion."""
    strategy = create_strategy()

    strategy._state_store.clear()  # type: ignore

    dates = list(strategy.calendar)[:20]  # First 20 dates

    for d in dates:
        state = strategy.compute_state(d)
        assert state is not None

    later_date = date.fromisoformat("2023-06-29")  # Not last date
    state = strategy.compute_state(later_date)

    assert state is not None
    cached_count = sum(
        1 for d in dates if strategy._state_store.get(d) is not None  # type: ignore
    )
    assert cached_count == len(dates), "Expected all sequential dates to be cached"


def test_recursion_error_handling():
    """Test that recursion handles errors gracefully (e.g., missing market data)."""
    strategy = create_strategy()

    target_date = date.fromisoformat("2023-01-10")

    state = strategy.compute_state(target_date)
    assert state is not None

    with pytest.raises(Exception):  # Should raise ScheduleError
        strategy.compute_state(date.fromisoformat("2023-01-01"))


def test_recursion_with_empty_cache():
    """Test recursion behavior when cache is empty."""
    strategy = create_strategy()

    strategy._state_store.clear()  # type: ignore

    target_date = date.fromisoformat("2023-03-31")

    calendar = strategy.calendar
    dates_list = list(calendar)
    seed_idx = dates_list.index(strategy.seed_date)
    target_idx = dates_list.index(target_date)
    expected_depth = target_idx - seed_idx

    assert expected_depth < sys.getrecursionlimit()

    state = strategy.compute_state(target_date)
    assert state is not None

    assert strategy._state_store.get(strategy.seed_date) is not None  # type: ignore


def test_recursion_limit_safety():
    """Test that recursion depth stays within Python's recursion limit."""
    strategy = create_strategy()

    recursion_limit = sys.getrecursionlimit()

    calendar = strategy.calendar
    dates_list = list(calendar)
    seed_idx = dates_list.index(strategy.seed_date)
    last_idx = len(dates_list) - 1
    max_depth = last_idx - seed_idx

    assert max_depth < recursion_limit, (
        f"Maximum recursion depth {max_depth} exceeds Python's limit {recursion_limit}. "
        "This could cause stack overflow."
    )

    strategy._state_store.clear()  # type: ignore
    test_date = dates_list[-2] if len(dates_list) > 1 else dates_list[-1]

    state = strategy.compute_state(test_date)
    assert state is not None


def test_recursion_overflow_scenario():
    """Test that recursion overflow raises RecursionError when depth exceeds limit."""
    # Save original limit
    original_limit = sys.getrecursionlimit()
    
    try:
        # Create market data with enough dates to potentially exceed a low recursion limit
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,ticker,close\n")
            # Create 60 dates (more than a low limit we'll set)
            # Start from 2023-01-02 (seed_date) and go forward
            for i in range(60):
                test_date = date.fromisoformat("2023-01-02") + date.resolution * i
                # Skip weekends - only add weekdays
                if test_date.weekday() < 5:  # Monday=0, Friday=4
                    f.write(f"{test_date},SPX,{4000 + i}\n")
                    f.write(f"{test_date},SX5E,{3700 + i}\n")
                    f.write(f"{test_date},HSI,{21000 + i}\n")
            temp_file = f.name
        
        try:
            # Load data with normal recursion limit (pandas needs it)
            md = MarketData(temp_file)
            strategy = EqualWeightStrategy(
                md=md,
                basket=["SPX", "SX5E", "HSI"],
                seed_date=date.fromisoformat("2023-01-02"),
                calendar=md.get_calendar(),
                initial_index_level=100,
            )
            
            strategy._state_store.clear()  # type: ignore
            
            # Get the last date in the calendar
            calendar = strategy.calendar
            dates_list = list(calendar)
            seed_idx = dates_list.index(strategy.seed_date)
            last_date = dates_list[-1]
            last_idx = dates_list.index(last_date)
            required_depth = last_idx - seed_idx
            
            # Set a low recursion limit AFTER data loading
            # Use a limit that's less than required depth but not too low
            # Need to leave enough room for Python's own operations (at least 30-40)
            test_limit = min(required_depth - 5, 50)  # Ensure it's less than required depth
            if test_limit < 30:  # Safety check - need room for Python's own operations
                test_limit = 30
            
            # Verify that required depth exceeds the test limit
            assert required_depth > test_limit, (
                f"Test setup error: required depth {required_depth} should exceed limit {test_limit}. "
                f"Try increasing the number of dates in the test CSV."
            )
            
            # Set recursion limit right before the call that should fail
            sys.setrecursionlimit(test_limit)
            
            # Attempting to compute the last date should raise RecursionError
            # because it requires recursion depth > test_limit
            # Use try-except instead of pytest.raises to avoid recursion issues with pytest itself
            recursion_error_raised = False
            try:
                strategy.compute_state(last_date)
                # If we get here, the recursion error wasn't raised
            except RecursionError:
                # This is the expected behavior - recursion limit was exceeded
                recursion_error_raised = True
            except Exception as e:
                # If we get a different exception, that's unexpected
                pytest.fail(f"Expected RecursionError, but got {type(e).__name__}: {e}")
            
            # Verify that RecursionError was actually raised
            assert recursion_error_raised, (
                "Expected RecursionError to be raised when recursion depth exceeds limit, "
                "but computation succeeded"
            )
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            # Restore original recursion limit
            sys.setrecursionlimit(original_limit)
    except Exception:
        # Ensure we restore the limit even if test fails
        sys.setrecursionlimit(original_limit)
        raise

