"""
Tests for runner module (get_states function).
"""

import pytest
from dataclasses import dataclass
from datetime import date
from typing import Optional, Any
from marketdata import MarketData
from rule import EqualWeightStrategy
from runner import get_states
from base import Strategy, StrategyState
from schedule import Schedule


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


def test_get_states_with_from_date():
    """Test get_states with explicit from_date."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-03")
    to_date = date.fromisoformat("2023-01-05")

    states = get_states(strategy, from_date, to_date)

    assert isinstance(states, dict)
    assert len(states) > 0
    assert from_date in states
    assert to_date in states

    # Verify all dates are in range
    for state_date in states.keys():
        assert from_date <= state_date <= to_date


def test_get_states_with_none_from_date():
    """Test get_states with from_date=None (should use seed_date)."""
    strategy = create_strategy()
    to_date = date.fromisoformat("2023-01-05")

    states = get_states(strategy, None, to_date)

    assert isinstance(states, dict)
    assert len(states) > 0
    assert strategy.seed_date in states
    assert to_date in states

    # Verify all dates are from seed_date onwards
    for state_date in states.keys():
        assert strategy.seed_date <= state_date <= to_date


def test_get_states_single_date():
    """Test get_states with single date range."""
    strategy = create_strategy()
    target_date = date.fromisoformat("2023-01-03")

    states = get_states(strategy, target_date, target_date)

    assert len(states) == 1
    assert target_date in states
    assert states[target_date].index_level > 0


def test_get_states_empty_range():
    """Test get_states with empty date range."""
    strategy = create_strategy()
    # Use dates that are definitely outside the calendar range
    from_date = date.fromisoformat("2024-01-01")  # Well after data range
    to_date = date.fromisoformat("2024-01-02")

    states = get_states(strategy, from_date, to_date)

    assert isinstance(states, dict)
    assert len(states) == 0


def test_get_states_seed_date_to_end():
    """Test get_states from seed_date to end of data."""
    strategy = create_strategy()
    to_date = date.fromisoformat("2023-06-29")

    states = get_states(strategy, None, to_date)

    assert len(states) > 0
    assert strategy.seed_date in states
    assert to_date in states

    # Verify states are computed for all dates in calendar
    calendar = strategy.resolve_dates(None, to_date)
    assert len(states) == len(calendar)


def test_get_states_returns_correct_states():
    """Test that get_states returns correct state objects."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-03")
    to_date = date.fromisoformat("2023-01-05")

    states = get_states(strategy, from_date, to_date)

    # Verify each state has required attributes
    for _, state in states.items():
        assert hasattr(state, "index_level")
        assert hasattr(state, "weights")
        assert hasattr(state, "returns")
        assert hasattr(state, "portfolio_return")
        assert isinstance(state.index_level, float)
        assert isinstance(state.weights, dict)
        assert state.index_level > 0


def test_get_states_consistency():
    """Test that get_states returns same results as individual compute_state calls."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-03")
    to_date = date.fromisoformat("2023-01-05")

    states_batch = get_states(strategy, from_date, to_date)

    # Compare with individual calls
    for state_date in states_batch.keys():
        state_individual = strategy.compute_state(state_date)
        assert states_batch[state_date].index_level == state_individual.index_level
        assert states_batch[state_date].weights == state_individual.weights


def test_get_states_uses_caching():
    """Test that get_states benefits from caching."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-05")

    # First call - populates cache
    states1 = get_states(strategy, from_date, to_date)

    # Second call - should use cache
    states2 = get_states(strategy, from_date, to_date)

    # Results should be identical
    assert len(states1) == len(states2)
    for date_key in states1.keys():
        assert states1[date_key].index_level == states2[date_key].index_level


def test_get_states_date_order():
    """Test that get_states returns dates in chronological order."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-10")

    states = get_states(strategy, from_date, to_date)

    dates = list(states.keys())
    assert dates == sorted(dates)  # Should be in chronological order


def test_get_states_with_updated_market_data():
    """Test that get_states reflects market data updates."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-05")

    # Get initial states
    states_before = get_states(strategy, from_date, to_date)
    levels_before = {d: s.index_level for d, s in states_before.items()}

    # Update market data
    update_date = date.fromisoformat("2023-01-03")
    original_price = strategy.md.get(update_date, "SPX")
    strategy.md.update(update_date, "SPX", original_price * 1.1)

    # Get states again - should reflect update
    states_after = get_states(strategy, from_date, to_date)
    levels_after = {d: s.index_level for d, s in states_after.items()}

    # Dates before update should be same
    assert (
        levels_after[date.fromisoformat("2023-01-02")]
        == levels_before[date.fromisoformat("2023-01-02")]
    )

    # Dates at and after update should be different
    assert levels_after[update_date] != levels_before[update_date]
    assert (
        levels_after[date.fromisoformat("2023-01-04")]
        != levels_before[date.fromisoformat("2023-01-04")]
    )
    assert (
        levels_after[date.fromisoformat("2023-01-05")]
        != levels_before[date.fromisoformat("2023-01-05")]
    )


def test_get_states_from_date_after_to_date():
    """Test get_states when from_date is after to_date."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-05")
    to_date = date.fromisoformat("2023-01-03")

    states = get_states(strategy, from_date, to_date)

    # Should return empty dict (invalid range)
    assert len(states) == 0


def test_get_states_large_range():
    """Test get_states with a large date range."""
    strategy = create_strategy()
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-06-29")

    states = get_states(strategy, from_date, to_date)

    assert len(states) > 100  # Should have many dates
    assert from_date in states
    assert to_date in states

    # Verify all states are valid
    for state in states.values():
        assert state.index_level > 0
        assert sum(state.weights.values()) == pytest.approx(1.0, rel=1e-6)  # type: ignore


# ========== Strategy Replaceability Tests ==========


@dataclass(frozen=True)
class MockStrategyState:
    """Mock state for testing strategy replaceability."""
    value: float
    multiplier: float
    description: str


@dataclass(frozen=True)
class MockStrategy(Strategy[MockStrategyState]):
    """
    Mock strategy implementation for testing replaceability.
    This strategy computes a simple value based on market data.
    """
    seed_date: date
    calendar: Schedule
    initial_value: float = 100.0
    multiplier: float = 1.5
    
    def resolve_dates(self, from_date: Optional[date], to_date: date) -> Schedule:
        """Resolve dates using the calendar."""
        if from_date is None:
            from_date = self.seed_date
        return self.calendar.sub_schedule(from_date, to_date)
    
    def compute_state(self, date: date) -> MockStrategyState:
        """Compute a simple mock state based on market data."""
        if date == self.seed_date:
            return MockStrategyState(
                value=self.initial_value,
                multiplier=self.multiplier,
                description=f"Initial state at {date}"
            )
        
        # For subsequent dates, use a simple calculation based on SPX price
        try:
            price = self.md.get(date, "SPX")
            # Simple transformation: value = initial * (price / base_price) * multiplier
            base_price = self.md.get(self.seed_date, "SPX")
            value = self.initial_value * (price / base_price) * self.multiplier
            
            return MockStrategyState(
                value=value,
                multiplier=self.multiplier,
                description=f"Computed state at {date}"
            )
        except Exception:
            # Fallback if asset not available
            return MockStrategyState(
                value=self.initial_value,
                multiplier=self.multiplier,
                description=f"Fallback state at {date}"
            )


def test_get_states_with_mock_strategy():
    """Test that get_states works with a mock strategy implementation."""
    md = MarketData("sample_prices.csv")
    mock_strategy = MockStrategy(
        md=md,
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_value=100.0,
        multiplier=1.5,
    )
    
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-05")
    
    states = get_states(mock_strategy, from_date, to_date)
    
    assert isinstance(states, dict)
    assert len(states) > 0
    assert from_date in states
    assert to_date in states
    
    # Verify all states are MockStrategyState instances
    for state in states.values():
        assert isinstance(state, MockStrategyState)
        assert hasattr(state, "value")
        assert hasattr(state, "multiplier")
        assert hasattr(state, "description")
        assert isinstance(state.value, float)
        assert state.value > 0


def test_get_states_strategy_replaceability():
    """Test that get_states works with different strategy implementations."""
    md = MarketData("sample_prices.csv")
    
    # Create two different strategy implementations
    equal_weight_strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )
    
    mock_strategy = MockStrategy(
        md=md,
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_value=100.0,
        multiplier=1.5,
    )
    
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-05")
    
    # Both strategies should work with get_states
    equal_weight_states = get_states(equal_weight_strategy, from_date, to_date)
    mock_states = get_states(mock_strategy, from_date, to_date)
    
    # Both should return dictionaries with same dates
    assert len(equal_weight_states) == len(mock_states)
    assert set(equal_weight_states.keys()) == set(mock_states.keys())
    
    # But states should be different types
    assert isinstance(list(equal_weight_states.values())[0], type(equal_weight_strategy.compute_state(from_date)))
    assert isinstance(list(mock_states.values())[0], MockStrategyState)


def test_strategy_interface_abstraction():
    """Test that the Strategy interface abstraction works correctly."""
    md = MarketData("sample_prices.csv")
    
    strategies = [
        EqualWeightStrategy(
            md=md,
            basket=["SPX", "SX5E", "HSI"],
            seed_date=date.fromisoformat("2023-01-02"),
            calendar=md.get_calendar(),
            initial_index_level=100,
        ),
        MockStrategy(
            md=md,
            seed_date=date.fromisoformat("2023-01-02"),
            calendar=md.get_calendar(),
            initial_value=100.0,
            multiplier=2.0,
        ),
    ]
    
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-05")
    
    # All strategies should implement the required interface
    for strategy in strategies:
        assert isinstance(strategy, Strategy)
        assert hasattr(strategy, "md")
        assert hasattr(strategy, "resolve_dates")
        assert hasattr(strategy, "compute_state")
        
        # All should work with get_states
        states = get_states(strategy, from_date, to_date)
        assert isinstance(states, dict)
        assert len(states) > 0
        
        # All should have resolve_dates that returns a Schedule
        schedule = strategy.resolve_dates(from_date, to_date)
        assert isinstance(schedule, Schedule)
        assert len(schedule) > 0


def test_strategy_swapping():
    """Test that strategies can be swapped and used interchangeably."""
    md = MarketData("sample_prices.csv")
    
    # Create a function that works with any strategy
    def compute_index_levels(strategy: Strategy[Any], from_date: date, to_date: date) -> dict[date, float]:
        """Generic function that works with any strategy."""
        states = get_states(strategy, from_date, to_date)
        # Extract index level or value depending on strategy type
        if hasattr(list(states.values())[0], "index_level"):
            return {d: getattr(s, "index_level") for d, s in states.items()}
        elif hasattr(list(states.values())[0], "value"):
            return {d: getattr(s, "value") for d, s in states.items()}
        else:
            raise ValueError("State type not recognized")
    
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-05")
    
    # Test with EqualWeightStrategy
    equal_weight_strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )
    equal_weight_levels = compute_index_levels(equal_weight_strategy, from_date, to_date)
    assert len(equal_weight_levels) > 0
    assert all(level > 0 for level in equal_weight_levels.values())
    
    # Test with MockStrategy
    mock_strategy = MockStrategy(
        md=md,
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_value=100.0,
        multiplier=1.5,
    )
    mock_levels = compute_index_levels(mock_strategy, from_date, to_date)
    assert len(mock_levels) > 0
    assert all(level > 0 for level in mock_levels.values())
    
    # Both should have same dates
    assert set(equal_weight_levels.keys()) == set(mock_levels.keys())


def test_mock_strategy_state_consistency():
    """Test that mock strategy produces consistent states."""
    md = MarketData("sample_prices.csv")
    mock_strategy = MockStrategy(
        md=md,
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_value=100.0,
        multiplier=1.5,
    )
    
    test_date = date.fromisoformat("2023-01-03")
    
    # Compute state multiple times - should be consistent
    state1 = mock_strategy.compute_state(test_date)
    state2 = mock_strategy.compute_state(test_date)
    
    assert state1.value == state2.value
    assert state1.multiplier == state2.multiplier
    assert state1.description == state2.description


def test_different_strategy_types_same_interface():
    """Test that different strategy types can be used through the same interface."""
    md = MarketData("sample_prices.csv")
    
    strategies = [
        EqualWeightStrategy(
            md=md,
            basket=["SPX", "SX5E", "HSI"],
            seed_date=date.fromisoformat("2023-01-02"),
            calendar=md.get_calendar(),
            initial_index_level=100,
        ),
        MockStrategy(
            md=md,
            seed_date=date.fromisoformat("2023-01-02"),
            calendar=md.get_calendar(),
            initial_value=100.0,
            multiplier=1.5,
        ),
    ]
    
    from_date = date.fromisoformat("2023-01-02")
    to_date = date.fromisoformat("2023-01-05")
    
    # All strategies should work through the same interface
    for strategy in strategies:
        # Test resolve_dates
        schedule = strategy.resolve_dates(from_date, to_date)
        assert isinstance(schedule, Schedule)
        assert from_date in schedule or schedule.sub_schedule(from_date, to_date)
        
        # Test compute_state
        state = strategy.compute_state(from_date)
        assert state is not None
        
        # Test get_states
        states = get_states(strategy, from_date, to_date)
        assert isinstance(states, dict)
        assert len(states) > 0
        assert from_date in states
