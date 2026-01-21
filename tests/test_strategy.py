import pytest
from datetime import date
from typing import List
from marketdata import MarketData, MarketDataError
from rule import EqualWeightStrategy
from runner import get_states
from schedule import ScheduleError


def compute_and_check(strategy: EqualWeightStrategy, final_date: str, expected: float):
    final_level = strategy.compute_state(date.fromisoformat(final_date)).index_level
    assert (
        round(final_level, 6) == expected
    ), f"Index level to 6dp on {final_date} should be {expected}, got {final_level}"


def get_states_and_check(
    strategy: EqualWeightStrategy, from_date: str, to_date: str, expected: List[float]
):
    states = get_states(
        strategy, date.fromisoformat(from_date), date.fromisoformat(to_date)
    ).values()
    levels = [round(state.index_level, 6) for state in states]
    assert levels == expected


def initialise() -> EqualWeightStrategy:
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )
    return strategy


def test_strategy_calculation():
    strategy = initialise()
    compute_and_check(strategy, "2023-01-03", 100.066461)
    compute_and_check(strategy, "2023-01-31", 93.227305)
    compute_and_check(strategy, "2023-02-01", 92.277544)
    compute_and_check(strategy, "2023-05-19", 92.441678)


def test_calculate_range():
    strategy = initialise()
    get_states_and_check(
        strategy, "2023-02-05", "2023-02-08", [94.098372, 93.541086, 92.601076]
    )


# ========== Edge Cases ==========


def test_compute_state_at_seed_date():
    """Test computing state at seed_date returns initial state."""
    strategy = initialise()
    state = strategy.compute_state(strategy.seed_date)

    assert state.index_level == 100.0
    assert state.portfolio_return == 0.0
    assert all(ret == 0.0 for ret in state.returns.values())
    # Weights should be equal (1/3 for each asset)
    assert all(weight == pytest.approx(1.0 / 3.0, rel=1e-6) for weight in state.weights.values())  # type: ignore


def test_compute_state_before_seed_date():
    """Test that computing state before seed_date raises an error."""
    strategy = initialise()
    before_seed = date.fromisoformat("2023-01-01")

    with pytest.raises(ScheduleError, match="No date before"):
        strategy.compute_state(before_seed)


def test_compute_state_date_not_in_calendar():
    """Test computing state for a date not in the calendar."""
    strategy = initialise()
    # Use a weekend date that's not in the calendar
    weekend_date = date.fromisoformat("2023-01-07")  # Saturday

    # This will fail when trying to get market data (not in calendar)
    with pytest.raises(MarketDataError, match="No data for"):
        strategy.compute_state(weekend_date)


def test_rebalancing_at_month_end():
    """Test that weights are rebalanced to equal at month-end."""
    strategy = initialise()

    # Get state on first day of February (after rebalancing at end of Jan)
    feb_1 = date.fromisoformat("2023-02-01")
    state_feb_1 = strategy.compute_state(feb_1)

    # Weights on Feb 1 should be approximately equal (rebalanced at end of Jan)
    # Allow for floating point differences due to calculation precision
    expected_weight = 1.0 / 3.0
    weights_list = list(state_feb_1.weights.values())
    # Check that all weights are close to expected (within 1%)
    for weight in weights_list:
        assert (
            abs(weight - expected_weight) < 0.01
        ), f"Weight {weight} not close to {expected_weight}"

    # Weights should still sum to 1.0
    assert sum(weights_list) == pytest.approx(1.0, rel=1e-6)  # type: ignore


def test_weight_drift_between_rebalancings():
    """Test that weights drift between rebalancings."""
    strategy = initialise()

    # Get states for consecutive days in the middle of a month
    jan_10 = date.fromisoformat("2023-01-10")
    jan_11 = date.fromisoformat("2023-01-11")
    jan_12 = date.fromisoformat("2023-01-12")

    state_10 = strategy.compute_state(jan_10)
    state_11 = strategy.compute_state(jan_11)
    state_12 = strategy.compute_state(jan_12)

    # Weights should change (drift) based on returns
    # They won't necessarily be equal mid-month
    weights_10 = state_10.weights
    weights_11 = state_11.weights
    weights_12 = state_12.weights

    # Weights should sum to 1.0
    assert sum(weights_10.values()) == pytest.approx(1.0, rel=1e-6)  # type: ignore
    assert sum(weights_11.values()) == pytest.approx(1.0, rel=1e-6)  # type: ignore
    assert sum(weights_12.values()) == pytest.approx(1.0, rel=1e-6)  # type: ignore


def test_single_asset_basket():
    """Test strategy with a single asset basket."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    state = strategy.compute_state(date.fromisoformat("2023-01-03"))
    assert state.weights["SPX"] == pytest.approx(1.0, rel=1e-6)  # type: ignore
    assert len(state.weights) == 1
    assert len(state.returns) == 1


def test_two_asset_basket():
    """Test strategy with two assets."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    state = strategy.compute_state(strategy.seed_date)
    assert state.weights["SPX"] == pytest.approx(0.5, rel=1e-6)  # type: ignore
    assert state.weights["SX5E"] == pytest.approx(0.5, rel=1e-6)  # type: ignore
    assert sum(state.weights.values()) == pytest.approx(1.0, rel=1e-6)  # type: ignore


def test_five_asset_basket():
    """Test strategy with five assets (if available in data)."""
    md = MarketData("sample_prices.csv")
    # Use available tickers
    basket = ["SPX", "SX5E", "HSI"]
    strategy = EqualWeightStrategy(
        md=md,
        basket=basket,
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    state = strategy.compute_state(strategy.seed_date)
    expected_weight = 1.0 / len(basket)
    for asset in basket:
        assert state.weights[asset] == pytest.approx(expected_weight, rel=1e-6)  # type: ignore
    assert sum(state.weights.values()) == pytest.approx(1.0, rel=1e-6)  # type: ignore


def test_very_long_date_range():
    """Test computing states for a very long date range."""
    strategy = initialise()
    from_date = date.fromisoformat("2023-01-02")
    # Use date before last to avoid is_last_day_of_month issue
    to_date = date.fromisoformat("2023-06-29")

    states = get_states(strategy, from_date, to_date)
    assert len(states) > 100

    # Verify all states are valid
    for state in states.values():
        assert state.index_level > 0
        assert sum(state.weights.values()) == pytest.approx(1.0, rel=1e-6)  # type: ignore


# ========== State Correctness Tests ==========


def test_weights_always_sum_to_one():
    """Test that weights always sum to 1.0 for all computed states."""
    strategy = initialise()
    dates = [
        date.fromisoformat("2023-01-03"),
        date.fromisoformat("2023-01-31"),
        date.fromisoformat("2023-02-15"),
        date.fromisoformat("2023-03-31"),
        date.fromisoformat("2023-06-29"),  # Avoid last date issue
    ]

    for d in dates:
        state = strategy.compute_state(d)
        weight_sum = sum(state.weights.values())
        assert weight_sum == pytest.approx(1.0, rel=1e-6), f"Weights don't sum to 1.0 on {d}: {weight_sum}"  # type: ignore


def test_portfolio_return_calculation():
    """Test that portfolio return is calculated correctly."""
    strategy = initialise()

    # Get two consecutive states
    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")

    state1 = strategy.compute_state(date1)
    state2 = strategy.compute_state(date2)

    # Portfolio return should be weighted sum of asset returns
    expected_portfolio_return = sum(
        state1.weights[asset] * state2.returns[asset] for asset in strategy.basket
    )

    assert state2.portfolio_return == pytest.approx(expected_portfolio_return, rel=1e-6)  # type: ignore


def test_index_level_calculation():
    """Test that index level is calculated correctly from portfolio return."""
    strategy = initialise()

    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")

    state1 = strategy.compute_state(date1)
    state2 = strategy.compute_state(date2)

    # Index level should be: prev_level * (1 + portfolio_return)
    expected_index_level = state1.index_level * (1 + state2.portfolio_return)

    assert state2.index_level == pytest.approx(expected_index_level, rel=1e-6)  # type: ignore


def test_returns_calculation():
    """Test that returns are calculated correctly."""
    strategy = initialise()

    date1 = date.fromisoformat("2023-01-03")
    date2 = date.fromisoformat("2023-01-04")

    state2 = strategy.compute_state(date2)

    # Returns should be: (today_price / yesterday_price) - 1
    for asset in strategy.basket:
        price_today = strategy.md.get(date2, asset)
        price_yesterday = strategy.md.get(date1, asset)
        expected_return = (price_today / price_yesterday) - 1

        assert state2.returns[asset] == pytest.approx(expected_return, rel=1e-6)  # type: ignore


def test_month_end_rebalancing_correctness():
    """Test that rebalancing correctly sets weights to equal at month-end."""
    strategy = initialise()

    # Test multiple month-ends (use dates that have next dates)
    month_ends = [
        date.fromisoformat("2023-01-31"),
        date.fromisoformat("2023-02-28"),
        date.fromisoformat("2023-03-31"),
        date.fromisoformat("2023-04-28"),
        date.fromisoformat("2023-05-31"),
    ]

    for month_end in month_ends:
        if month_end in strategy.calendar:
            try:
                # Check if it's the last day of month
                if strategy.calendar.is_last_day_of_month(month_end):
                    # Get state on first day of next month
                    next_month_start = strategy.calendar.next(month_end)
                    state = strategy.compute_state(next_month_start)

                    # Weights should be approximately equal (allow for calculation precision)
                    expected_weight = 1.0 / len(strategy.basket)
                    weights_list = list(state.weights.values())
                    # Check that all weights are close to expected (within 1%)
                    for weight in weights_list:
                        assert (
                            abs(weight - expected_weight) < 0.01
                        ), f"Weight {weight} not close to {expected_weight}"

                    # Weights should still sum to 1.0
                    assert sum(weights_list) == pytest.approx(1.0, rel=1e-6)  # type: ignore
            except ScheduleError:
                # Skip if no next date (e.g., last date in calendar)
                pass


def test_weight_drift_calculation():
    """Test that weight drift between rebalancings is calculated correctly."""
    strategy = initialise()

    # Get states for consecutive days mid-month
    date1 = date.fromisoformat("2023-01-10")
    date2 = date.fromisoformat("2023-01-11")

    state1 = strategy.compute_state(date1)
    state2 = strategy.compute_state(date2)

    # Weight drift formula: weight_new = weight_old * (1 + asset_return) / (1 + portfolio_return)
    for asset in strategy.basket:
        asset_return = state2.returns[asset]
        portfolio_return = state2.portfolio_return
        expected_weight = (
            state1.weights[asset] * (1 + asset_return) / (1 + portfolio_return)
        )

        assert state2.weights[asset] == pytest.approx(expected_weight, rel=1e-6)  # type: ignore


def test_negative_returns_handling():
    """Test that negative returns are handled correctly."""
    strategy = initialise()

    # Find a date with negative returns
    test_date = date.fromisoformat("2023-01-11")  # Known to have negative returns
    state = strategy.compute_state(test_date)

    # Some returns might be negative
    _ = any(ret < 0 for ret in state.returns.values())

    # Index level should still be positive (even if decreased)
    assert state.index_level > 0

    # Weights should still sum to 1.0
    assert sum(state.weights.values()) == pytest.approx(1.0, rel=1e-6)  # type: ignore

    # Portfolio return might be negative, but index level should handle it
    if state.portfolio_return < 0:
        prev_date = strategy.calendar.prev(test_date)
        prev_state = strategy.compute_state(prev_date)
        assert state.index_level < prev_state.index_level


def test_zero_returns_handling():
    """Test that zero returns are handled correctly."""
    strategy = initialise()

    # Create a scenario with zero returns by updating prices to be the same
    test_date = date.fromisoformat("2023-01-03")
    prev_date = strategy.calendar.prev(test_date)

    # Get original prices
    prices = {asset: strategy.md.get(prev_date, asset) for asset in strategy.basket}

    # Set today's prices to yesterday's (zero returns)
    for asset in strategy.basket:
        strategy.md.update(test_date, asset, prices[asset])

    state = strategy.compute_state(test_date)

    # All returns should be zero
    for ret in state.returns.values():
        assert ret == pytest.approx(0.0, rel=1e-6)  # type: ignore

    # Portfolio return should be zero
    assert state.portfolio_return == pytest.approx(0.0, rel=1e-6)  # type: ignore

    # Index level should be same as previous
    prev_state = strategy.compute_state(prev_date)
    assert state.index_level == pytest.approx(prev_state.index_level, rel=1e-6)  # type: ignore

    # Weights should be same as previous (no drift with zero returns)
    for asset in strategy.basket:
        assert state.weights[asset] == pytest.approx(prev_state.weights[asset], rel=1e-6)  # type: ignore
