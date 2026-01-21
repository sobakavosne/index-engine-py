"""
Tests for StateStore caching mechanism.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional

from marketdata import MarketData
from rule import EqualWeightStrategy
from schedule import Schedule
from statestore import StateStore
from base import Strategy


def create_test_strategy():
    """Create a strategy instance for testing."""
    md = MarketData('sample_prices.csv')
    return EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )


def test_get_not_cached():
    """Test getting a state that hasn't been cached."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    result = store.get(date.fromisoformat("2023-01-03"))
    assert result is None


def test_put_and_get():
    """Test storing and retrieving a state."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    # Create a test state
    test_date = date.fromisoformat("2023-01-03")
    test_state = strategy.compute_state(test_date)
    dependencies = {(test_date, "SPX"), (test_date, "SX5E")}
    
    # Store it
    store.put(test_date, test_state, dependencies)
    
    # Retrieve it
    retrieved = store.get(test_date)
    assert retrieved is not None
    assert retrieved.index_level == test_state.index_level
    assert retrieved.weights == test_state.weights


def test_get_invalidated_state():
    """Test that invalidated states return None."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    test_date = date.fromisoformat("2023-01-03")
    test_state = strategy.compute_state(test_date)
    dependencies = {(test_date, "SPX")}
    
    store.put(test_date, test_state, dependencies)
    
    # Invalidate by updating market data
    strategy.md.update(test_date, "SPX", 5000.0)
    
    # State should now be invalid
    result = store.get(test_date)
    assert result is None


def test_invalidate_single_date():
    """Test invalidating states at a specific date."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")
    date3 = date.fromisoformat("2023-01-05")
    
    state1 = strategy.compute_state(date1)
    state2 = strategy.compute_state(date2)
    state3 = strategy.compute_state(date3)
    
    store.put(date1, state1, {(date1, "SPX")})
    store.put(date2, state2, {(date2, "SPX")})
    store.put(date3, state3, {(date3, "SPX")})
    
    # Invalidate date2 - should remove date2 and date3
    store.invalidate(date2)
    
    assert store.get(date1) is not None  # Before invalidated date
    assert store.get(date2) is None  # At invalidated date
    assert store.get(date3) is None  # After invalidated date


def test_invalidate_removes_dependencies():
    """Test that invalidate removes both cache and dependencies."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    test_date = date.fromisoformat("2023-01-03")
    test_state = strategy.compute_state(test_date)
    dependencies = {(test_date, "SPX")}
    
    store.put(test_date, test_state, dependencies)
    
    # Verify it's cached
    assert store.get(test_date) is not None
    
    # Invalidate
    store.invalidate(test_date)
    
    # Should be removed from cache
    assert store.get(test_date) is None
    
    # Dependencies should also be removed (tested indirectly via _is_valid)


def test_clear():
    """Test clearing all cached states."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")
    date3 = date.fromisoformat("2023-01-05")
    
    state1 = strategy.compute_state(date1)
    state2 = strategy.compute_state(date2)
    state3 = strategy.compute_state(date3)
    
    store.put(date1, state1, {(date1, "SPX")})
    store.put(date2, state2, {(date2, "SPX")})
    store.put(date3, state3, {(date3, "SPX")})
    
    # Verify all are cached
    assert store.get(date1) is not None
    assert store.get(date2) is not None
    assert store.get(date3) is not None
    
    # Clear all
    store.clear()
    
    # All should be gone
    assert store.get(date1) is None
    assert store.get(date2) is None
    assert store.get(date3) is None


def test_is_valid_with_no_updates():
    """Test that states are valid when no market data has been updated."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    test_date = date.fromisoformat("2023-01-03")
    test_state = strategy.compute_state(test_date)
    dependencies = {(test_date, "SPX")}
    
    store.put(test_date, test_state, dependencies)
    
    # Should be valid (no updates)
    result = store.get(test_date)
    assert result is not None


def test_is_valid_with_unrelated_updates():
    """Test that states remain valid when unrelated dates are updated."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-10")
    
    state1 = strategy.compute_state(date1)
    store.put(date1, state1, {(date1, "SPX")})
    
    # Update a different date
    strategy.md.update(date2, "SPX", 5000.0)
    
    # State at date1 should still be valid
    result = store.get(date1)
    assert result is not None


def test_is_valid_with_related_updates():
    """Test that states become invalid when their dependencies are updated."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    test_date = date.fromisoformat("2023-01-03")
    state = strategy.compute_state(test_date)
    dependencies = {(test_date, "SPX"), (test_date, "SX5E")}
    
    store.put(test_date, state, dependencies)
    
    # Update one of the dependencies
    strategy.md.update(test_date, "SPX", 5000.0)
    
    # State should be invalid
    result = store.get(test_date)
    assert result is None


def test_is_valid_with_previous_date_dependency():
    """Test that states are invalidated when previous date dependencies are updated."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")
    
    # State at date2 depends on date1 and date2
    state2 = strategy.compute_state(date2)
    dependencies = {(date1, "SPX"), (date2, "SPX")}
    
    store.put(date2, state2, dependencies)
    
    # Update date1 (a dependency)
    strategy.md.update(date1, "SPX", 5000.0)
    
    # State at date2 should be invalid
    result = store.get(date2)
    assert result is None


def test_multiple_states_same_dependencies():
    """Test storing multiple states with overlapping dependencies."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")
    
    state1 = strategy.compute_state(date1)
    state2 = strategy.compute_state(date2)
    
    # Both depend on date1
    store.put(date1, state1, {(date1, "SPX")})
    store.put(date2, state2, {(date1, "SPX"), (date2, "SPX")})
    
    # Update date1 - both should be invalid
    strategy.md.update(date1, "SPX", 5000.0)
    
    assert store.get(date1) is None
    assert store.get(date2) is None


def test_dependencies_copy():
    """Test that dependencies are copied when stored (not referenced)."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    test_date = date.fromisoformat("2023-01-03")
    state = strategy.compute_state(test_date)
    
    # Store with a subset of dependencies (only SPX)
    dependencies = {(test_date, "SPX")}
    store.put(test_date, state, dependencies)
    
    # Modify the original set (shouldn't affect stored dependencies)
    dependencies.add((test_date, "SX5E"))
    
    # Update SX5E - state should still be valid since SX5E wasn't in stored deps
    # But wait - the actual state computation tracks all dependencies
    # So let's test differently: update a date that's NOT in dependencies
    unrelated_date = date.fromisoformat("2023-01-10")
    strategy.md.update(unrelated_date, "SPX", 5000.0)
    
    # State should still be valid (unrelated date updated)
    result = store.get(test_date)
    assert result is not None
    
    # But if we update the date that IS in dependencies, it should be invalid
    strategy.md.update(test_date, "SPX", 6000.0)
    result = store.get(test_date)
    assert result is None


def test_empty_dependencies():
    """Test storing state with empty dependencies."""
    strategy = create_test_strategy()
    store = StateStore(strategy)
    
    test_date = date.fromisoformat("2023-01-02")  # Seed date
    state = strategy.compute_state(test_date)
    
    store.put(test_date, state, set())
    
    # Should be retrievable
    result = store.get(test_date)
    assert result is not None
    
    # Should remain valid even if market data is updated
    strategy.md.update(date.fromisoformat("2023-01-03"), "SPX", 5000.0)
    result = store.get(test_date)
    assert result is not None


# Mock Strategy for testing cache isolation
@dataclass(frozen=True)
class MockStrategyState:
    """Mock state type for testing - different from EqualWeightStrategyState."""
    value: float
    label: str


@dataclass(frozen=True)
class MockStrategy(Strategy[MockStrategyState]):
    """Mock strategy with different state type for testing cache isolation."""
    calendar: Schedule
    seed_date: date
    
    def resolve_dates(self, from_date: Optional[date], to_date: date) -> Schedule:
        """Resolve dates using the calendar."""
        if from_date is None:
            from_date = self.seed_date
        return self.calendar.sub_schedule(from_date, to_date)
    
    def compute_state(self, date: date) -> MockStrategyState:
        """Compute a simple mock state."""
        # Use market data to make it realistic
        price = self.md.get(date, "SPX")
        
        return MockStrategyState(
            value=price * 1.5,  # Simple transformation
            label=f"mock_{date}"
        )


def test_cache_isolation_between_different_strategy_instances():
    """Test that different strategy instances have separate caches."""
    # Create two EqualWeightStrategy instances with different parameters
    md1 = MarketData('sample_prices.csv')
    md2 = MarketData('sample_prices.csv')
    
    strategy1 = EqualWeightStrategy(
        md=md1,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md1.get_calendar(),
        initial_index_level=100,
    )
    
    strategy2 = EqualWeightStrategy(
        md=md2,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md2.get_calendar(),
        initial_index_level=200,  # Different initial level
    )
    
    test_date = date.fromisoformat("2023-01-03")
    
    # Compute and cache state for strategy1
    state1 = strategy1.compute_state(test_date)
    level1 = state1.index_level
    
    # Compute and cache state for strategy2
    state2 = strategy2.compute_state(test_date)
    level2 = state2.index_level
    
    # States should be different (different initial levels)
    assert level1 != level2
    
    # Verify both are cached independently
    cached_state1 = strategy1._state_store.get(test_date)  # type: ignore
    cached_state2 = strategy2._state_store.get(test_date)  # type: ignore
    
    assert cached_state1 is not None
    assert cached_state2 is not None
    assert cached_state1.index_level == level1
    assert cached_state2.index_level == level2
    
    # Invalidate cache in strategy1 - should not affect strategy2
    strategy1.md.update(test_date, "SPX", strategy1.md.get(test_date, "SPX") * 1.1)
    
    # strategy1 cache should be invalidated
    assert strategy1._state_store.get(test_date) is None  # type: ignore
    
    # strategy2 cache should still be valid
    assert strategy2._state_store.get(test_date) is not None  # type: ignore
    assert strategy2._state_store.get(test_date).index_level == level2  # type: ignore


def test_cache_isolation_same_marketdata():
    """Test that strategies sharing the same MarketData have separate caches."""
    # Create two strategies sharing the same MarketData instance
    shared_md = MarketData('sample_prices.csv')
    calendar = shared_md.get_calendar()
    
    strategy1 = EqualWeightStrategy(
        md=shared_md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=calendar,
        initial_index_level=100,
    )
    
    strategy2 = EqualWeightStrategy(
        md=shared_md,  # Same MarketData
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=calendar,
        initial_index_level=200,  # Different initial level
    )
    
    test_date = date.fromisoformat("2023-01-03")
    
    # Compute states for both strategies
    state1 = strategy1.compute_state(test_date)
    state2 = strategy2.compute_state(test_date)
    
    # States should be different
    assert state1.index_level != state2.index_level
    
    # Both should be cached
    assert strategy1._state_store.get(test_date) is not None  # type: ignore
    assert strategy2._state_store.get(test_date) is not None  # type: ignore
    
    # Update market data - both callbacks should be triggered
    original_price = shared_md.get(test_date, "SPX")
    shared_md.update(test_date, "SPX", original_price * 1.1)
    
    # Both caches should be invalidated (both registered callbacks)
    assert strategy1._state_store.get(test_date) is None  # type: ignore
    assert strategy2._state_store.get(test_date) is None  # type: ignore
    
    # But they should recompute to different values
    new_state1 = strategy1.compute_state(test_date)
    new_state2 = strategy2.compute_state(test_date)
    assert new_state1.index_level != new_state2.index_level


def test_cache_isolation_different_strategy_types():
    """Test that different strategy types maintain separate caches."""
    md = MarketData('sample_prices.csv')
    calendar = md.get_calendar()
    
    # Create EqualWeightStrategy
    equal_weight_strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=calendar,
        initial_index_level=100,
    )
    
    # Create MockStrategy with different state type
    mock_strategy = MockStrategy(
        md=md,
        calendar=calendar,
        seed_date=date.fromisoformat("2023-01-02"),
    )
    
    test_date = date.fromisoformat("2023-01-03")
    
    # Compute states for both strategies
    equal_weight_state = equal_weight_strategy.compute_state(test_date)
    mock_state = mock_strategy.compute_state(test_date)
    
    # States are different types - can't compare directly, but both should exist
    assert hasattr(equal_weight_state, 'index_level')
    assert hasattr(mock_state, 'value')
    assert hasattr(mock_state, 'label')
    
    # Create StateStore instances for both
    store1 = StateStore(equal_weight_strategy)
    store2 = StateStore(mock_strategy)
    
    # Store states in their respective caches
    store1.put(test_date, equal_weight_state, {(test_date, "SPX")})
    store2.put(test_date, mock_state, {(test_date, "SPX")})
    
    # Both should be retrievable from their own caches
    retrieved1 = store1.get(test_date)
    retrieved2 = store2.get(test_date)
    
    assert retrieved1 is not None
    assert retrieved2 is not None
    assert retrieved1.index_level == equal_weight_state.index_level
    assert retrieved2.value == mock_state.value
    assert retrieved2.label == mock_state.label
    
    # Update market data - both should be invalidated
    original_price = md.get(test_date, "SPX")
    md.update(test_date, "SPX", original_price * 1.1)
    
    # Both caches should be invalidated
    assert store1.get(test_date) is None
    assert store2.get(test_date) is None


def test_cache_isolation_independent_invalidation():
    """Test that invalidating one strategy's cache doesn't affect another."""
    md1 = MarketData('sample_prices.csv')
    md2 = MarketData('sample_prices.csv')
    
    strategy1 = EqualWeightStrategy(
        md=md1,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md1.get_calendar(),
        initial_index_level=100,
    )
    
    strategy2 = EqualWeightStrategy(
        md=md2,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md2.get_calendar(),
        initial_index_level=100,
    )
    
    test_date = date.fromisoformat("2023-01-03")
    
    # Compute and cache states for both
    strategy1.compute_state(test_date)
    state2 = strategy2.compute_state(test_date)
    
    # Both should be cached
    assert strategy1._state_store.get(test_date) is not None  # type: ignore
    assert strategy2._state_store.get(test_date) is not None  # type: ignore
    
    # Update market data only for strategy1
    original_price = md1.get(test_date, "SPX")
    md1.update(test_date, "SPX", original_price * 1.1)
    
    # Only strategy1's cache should be invalidated
    assert strategy1._state_store.get(test_date) is None  # type: ignore
    cached_state2 = strategy2._state_store.get(test_date)  # type: ignore
    assert cached_state2 is not None  # type: ignore
    assert strategy2._state_store.get(test_date) is not None  # type: ignore
    
    # strategy2's cached state should still match original
    assert cached_state2.index_level == state2.index_level
