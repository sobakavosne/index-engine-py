"""
Integration tests for cache invalidation when market data is updated.
"""
from datetime import date
from marketdata import MarketData
from rule import EqualWeightStrategy
from runner import get_states


def create_strategy():
    """Create a strategy instance for testing."""
    md = MarketData('sample_prices.csv')
    return EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )


def test_cache_invalidation_on_update():
    """Test that cache is invalidated when market data is updated."""
    strategy = create_strategy()
    
    # Compute state for a date
    target_date = date.fromisoformat("2023-01-03")
    state_before = strategy.compute_state(target_date)
    original_level = state_before.index_level
    
    # Verify it's cached (second call should be fast)
    state_cached = strategy.compute_state(target_date)
    assert state_cached.index_level == original_level
    
    # Update market data for that date
    original_price = strategy.md.get(target_date, "SPX")
    strategy.md.update(target_date, "SPX", original_price * 1.1)  # 10% increase
    
    # State should be recomputed (cache invalidated)
    state_after = strategy.compute_state(target_date)
    assert state_after.index_level != original_level
    assert state_after.index_level > original_level  # Price went up


def test_partial_invalidation():
    """Test that only affected dates are invalidated."""
    strategy = create_strategy()
    
    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")
    date3 = date.fromisoformat("2023-01-10")
    
    # Compute states for multiple dates
    state1_before = strategy.compute_state(date1)
    state2_before = strategy.compute_state(date2)
    state3_before = strategy.compute_state(date3)
    
    # Update date2
    original_price = strategy.md.get(date2, "SPX")
    strategy.md.update(date2, "SPX", original_price * 1.1)
    
    # date1 should still be cached (before update)
    state1_after = strategy.compute_state(date1)
    assert state1_after.index_level == state1_before.index_level
    
    # date2 and date3 should be recomputed (date2 directly, date3 depends on it)
    state2_after = strategy.compute_state(date2)
    state3_after = strategy.compute_state(date3)
    assert state2_after.index_level != state2_before.index_level
    assert state3_after.index_level != state3_before.index_level


def test_invalidation_cascade():
    """Test that invalidating an early date invalidates all later dates."""
    strategy = create_strategy()
    
    dates = [
        date.fromisoformat("2023-01-03"),
        date.fromisoformat("2023-01-04"),
        date.fromisoformat("2023-01-05"),
        date.fromisoformat("2023-01-06"),
    ]
    
    # Compute all states
    states_before = {d: strategy.compute_state(d) for d in dates}
    
    # Update the first date
    original_price = strategy.md.get(dates[0], "SPX")
    strategy.md.update(dates[0], "SPX", original_price * 1.1)
    
    # All dates should be recomputed (they all depend on earlier dates)
    for d in dates:
        state_after = strategy.compute_state(d)
        assert state_after.index_level != states_before[d].index_level


def test_multiple_updates_same_date():
    """Test multiple updates to the same date."""
    strategy = create_strategy()
    
    target_date = date.fromisoformat("2023-01-03")
    
    # Initial computation
    state1 = strategy.compute_state(target_date)
    level1 = state1.index_level
    
    # First update
    original_price = strategy.md.get(target_date, "SPX")
    strategy.md.update(target_date, "SPX", original_price * 1.1)
    state2 = strategy.compute_state(target_date)
    level2 = state2.index_level
    assert level2 != level1
    
    # Second update
    strategy.md.update(target_date, "SPX", original_price * 1.2)
    state3 = strategy.compute_state(target_date)
    level3 = state3.index_level
    assert level3 != level2
    assert level3 != level1


def test_update_different_ticker():
    """Test that updating one ticker invalidates states correctly."""
    strategy = create_strategy()
    
    target_date = date.fromisoformat("2023-01-03")
    
    # Compute initial state
    state_before = strategy.compute_state(target_date)
    original_level = state_before.index_level
    
    # Update SPX only
    original_spx = strategy.md.get(target_date, "SPX")
    strategy.md.update(target_date, "SPX", original_spx * 1.1)
    
    # State should be recomputed
    state_after = strategy.compute_state(target_date)
    assert state_after.index_level != original_level
    
    # Update SX5E (different ticker)
    original_sx5e = strategy.md.get(target_date, "SX5E")
    strategy.md.update(target_date, "SX5E", original_sx5e * 1.1)
    
    # State should be recomputed again
    state_after2 = strategy.compute_state(target_date)
    assert state_after2.index_level != state_after.index_level


def test_get_states_with_invalidation():
    """Test get_states() works correctly with cache invalidation."""
    strategy = create_strategy()
    
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-05")
    
    # Get initial states
    states_before = get_states(strategy, from_date, to_date)
    levels_before = {d: s.index_level for d, s in states_before.items()}
    
    # Update a date in the middle
    update_date = date.fromisoformat("2023-01-03")
    original_price = strategy.md.get(update_date, "SPX")
    strategy.md.update(update_date, "SPX", original_price * 1.1)
    
    # Get states again - should be recomputed
    states_after = get_states(strategy, from_date, to_date)
    levels_after = {d: s.index_level for d, s in states_after.items()}
    
    # Date before update should be same
    assert levels_after[date.fromisoformat("2023-01-02")] == levels_before[date.fromisoformat("2023-01-02")]
    
    # Dates at and after update should be different
    assert levels_after[update_date] != levels_before[update_date]
    assert levels_after[date.fromisoformat("2023-01-04")] != levels_before[date.fromisoformat("2023-01-04")]
    assert levels_after[date.fromisoformat("2023-01-05")] != levels_before[date.fromisoformat("2023-01-05")]


def test_callback_registration():
    """Test that strategy automatically registers invalidation callback."""
    strategy = create_strategy()
    
    # Strategy should have registered a callback
    # (tested indirectly - cache should be invalidated on update)
    target_date = date.fromisoformat("2023-01-03")
    state_before = strategy.compute_state(target_date)
    
    # Update market data
    original_price = strategy.md.get(target_date, "SPX")
    strategy.md.update(target_date, "SPX", original_price * 1.1)
    
    # Cache should be invalidated (callback was called)
    state_after = strategy.compute_state(target_date)
    assert state_after.index_level != state_before.index_level


def test_update_before_computation():
    """Test updating market data before computing states."""
    strategy = create_strategy()
    
    target_date = date.fromisoformat("2023-01-03")
    
    # Update before computing
    original_price = strategy.md.get(target_date, "SPX")
    strategy.md.update(target_date, "SPX", original_price * 1.1)
    
    # Compute state - should use updated price
    state = strategy.compute_state(target_date)
    
    # Verify it's different from what it would be with original price
    # (by comparing with a fresh strategy)
    strategy2 = create_strategy()
    state2 = strategy2.compute_state(target_date)
    assert state.index_level != state2.index_level


def test_clear_updated_dates():
    """Test that clearing updated dates doesn't affect cache validity."""
    strategy = create_strategy()
    
    target_date = date.fromisoformat("2023-01-03")
    
    # Compute and cache state
    state1 = strategy.compute_state(target_date)
    
    # Update market data
    original_price = strategy.md.get(target_date, "SPX")
    strategy.md.update(target_date, "SPX", original_price * 1.1)
    
    # State should be invalidated
    state2 = strategy.compute_state(target_date)
    assert state2.index_level != state1.index_level
    
    # Clear updated dates tracking
    strategy.md.clear_updated_dates()
    
    # State should still be invalid (cache was already cleared)
    # But if we cache it again, it should be valid
    state3 = strategy.compute_state(target_date)
    # Now cached with new data
    
    # Update again
    strategy.md.update(target_date, "SPX", original_price * 1.2)
    
    # Should be invalidated again
    state4 = strategy.compute_state(target_date)
    assert state4.index_level != state3.index_level

