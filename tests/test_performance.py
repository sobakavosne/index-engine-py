"""
Performance tests to compare cached vs uncached strategy computation.
"""
import time

from datetime import date
from typing import Tuple
from marketdata import MarketData
from rule import EqualWeightStrategy
from runner import get_states


def create_strategy() -> EqualWeightStrategy:
    """Create a strategy instance for testing."""
    md = MarketData('sample_prices.csv')
    return EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )


def time_computation(strategy: EqualWeightStrategy, from_date: date, to_date: date) -> Tuple[float, int]:
    """Time the computation of states for a date range."""
    start = time.perf_counter()
    states = get_states(strategy, from_date, to_date)
    end = time.perf_counter()
    return end - start, len(states)


def test_cached_performance():
    """Test that caching improves performance on repeated computations."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-06-29")
    
    time_cold, num_states = time_computation(strategy, from_date, to_date)
    
    time_warm, _ = time_computation(strategy, from_date, to_date)
    
    states = get_states(strategy, from_date, to_date)
    assert len(states) == num_states
    assert states[to_date].index_level > 0
    
    speedup = time_cold / time_warm if time_warm > 0 else float('inf')
    assert speedup >= 1.5, f"Expected cache speedup >= 1.5x, got {speedup:.2f}x"
    
    print(f"\nPerformance Test Results:")
    print(f"  Cold cache (first run): {time_cold*1000:.2f}ms")
    print(f"  Warm cache (second run): {time_warm*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x faster with cache")


def test_uncached_vs_cached_single_date():
    """Test performance difference when computing a single date multiple times."""
    strategy = create_strategy()
    target_date = date.fromisoformat("2023-06-29")
    
    start = time.perf_counter()
    state1 = strategy.compute_state(target_date)
    time_first = time.perf_counter() - start
    
    start = time.perf_counter()
    state2 = strategy.compute_state(target_date)
    time_cached = time.perf_counter() - start
    
    assert state1.index_level == state2.index_level
    
    speedup = time_first / time_cached if time_cached > 0 else float('inf')
    assert speedup >= 10, f"Expected cache speedup >= 10x for single date, got {speedup:.2f}x"
    
    print(f"\nSingle Date Cache Test:")
    print(f"  First computation: {time_first*1000:.3f}ms")
    print(f"  Cached computation: {time_cached*1000:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x faster with cache")


def test_uncached_performance():
    """Test performance without cache by clearing it between computations."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-06-29")
    
    time_cached, num_states = time_computation(strategy, from_date, to_date)
    
    strategy._state_store.clear()  # pyright: ignore[reportPrivateUsage]
    time_uncached, _ = time_computation(strategy, from_date, to_date)
    
    states = get_states(strategy, from_date, to_date)
    assert len(states) == num_states
    
    print(f"\nCached vs Uncached Comparison:")
    print(f"  With cache: {time_cached*1000:.2f}ms")
    print(f"  Without cache: {time_uncached*1000:.2f}ms")
    print(f"  Cache benefit: {((time_uncached - time_cached) / time_uncached * 100):.1f}% faster")


def test_sequential_vs_random_access():
    """Test that sequential access benefits more from cache than random access."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-06-29")
    
    strategy._state_store.clear()  # pyright: ignore[reportPrivateUsage]
    time_sequential, _ = time_computation(strategy, from_date, to_date)

    strategy._state_store.clear()  # pyright: ignore[reportPrivateUsage]
    schedule = strategy.resolve_dates(from_date, to_date)
    dates_list = list(schedule)
    
    start = time.perf_counter()
    for d in reversed(dates_list):
        strategy.compute_state(d)
    time_random = time.perf_counter() - start
    
    print(f"\nAccess Pattern Comparison:")
    print(f"  Sequential access: {time_sequential*1000:.2f}ms")
    print(f"  Random access: {time_random*1000:.2f}ms")
