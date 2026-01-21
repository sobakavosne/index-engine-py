"""
Error handling tests for strategy and market data operations.
"""

import pytest
import tempfile
import os
from datetime import date
from marketdata import MarketData, MarketDataError
from rule import EqualWeightStrategy
from runner import get_states
from schedule import ScheduleError


def test_compute_state_before_seed_date():
    """Test that computing state before seed_date raises ScheduleError."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    with pytest.raises(ScheduleError, match="No date before"):
        strategy.compute_state(date.fromisoformat("2023-01-01"))


def test_compute_state_date_not_in_calendar():
    """Test that computing state for date not in calendar raises error."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    # Try a date that's not in the calendar (e.g., weekend)
    with pytest.raises(MarketDataError, match="No data for"):
        strategy.compute_state(date.fromisoformat("2023-01-07"))


def test_get_states_date_outside_range():
    """Test get_states with dates outside data range."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    # Dates before seed_date - returns empty schedule (no error)
    states = get_states(
        strategy, date.fromisoformat("2022-12-01"), date.fromisoformat("2022-12-31")
    )
    assert len(states) == 0


def test_market_data_missing_ticker():
    """Test error when basket contains ticker not in market data."""
    md = MarketData("sample_prices.csv")

    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    # The error will occur during compute_state when trying to get price
    with pytest.raises(MarketDataError, match="No data for 'INVALID' on"):
        strategy.md.get(date.fromisoformat("2023-01-02"), "INVALID")


def test_market_data_missing_date():
    """Test error when trying to get price for date not in data."""
    md = MarketData("sample_prices.csv")

    with pytest.raises(MarketDataError, match="No data for"):
        md.get(date.fromisoformat("2020-01-01"), "SPX")


def test_strategy_with_empty_basket():
    """Test strategy initialization with empty basket."""
    md = MarketData("sample_prices.csv")

    # Empty basket - strategy can be created
    strategy = EqualWeightStrategy(
        md=md,
        basket=[],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    # Computing state at seed_date works (returns empty dicts)
    state = strategy.compute_state(strategy.seed_date)
    assert len(state.weights) == 0
    assert len(state.returns) == 0
    assert state.portfolio_return == 0.0
    assert state.index_level == 100.0

    # Computing a later date also works (empty basket means no returns to calculate)
    state2 = strategy.compute_state(date.fromisoformat("2023-01-03"))
    assert len(state2.weights) == 0
    assert len(state2.returns) == 0
    assert state2.portfolio_return == 0.0
    # Index level should remain the same (no portfolio return)
    assert state2.index_level == state.index_level


def test_invalid_initial_index_level_zero():
    """Test strategy with zero initial index level."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=0,
    )

    # Should work, but index level starts at 0
    state = strategy.compute_state(strategy.seed_date)
    assert state.index_level == 0.0


def test_invalid_initial_index_level_negative():
    """Test strategy with negative initial index level."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=-100,
    )

    state = strategy.compute_state(strategy.seed_date)
    assert state.index_level == -100.0


def test_resolve_dates_invalid_range():
    """Test resolve_dates with invalid date range."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    schedule = strategy.resolve_dates(
        date.fromisoformat("2023-01-10"), date.fromisoformat("2023-01-05")
    )
    assert len(schedule) == 0


def test_market_data_invalid_file_format():
    """Test error when CSV has invalid format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("wrong,columns\n")
        f.write("data,here\n")
        temp_file = f.name

    try:
        with pytest.raises(MarketDataError):
            MarketData(temp_file)
    finally:
        os.unlink(temp_file)


def test_market_data_missing_columns():
    """Test error when CSV is missing required columns."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date,ticker\n")  # Missing 'close' column
        f.write("2023-01-02,SPX\n")
        temp_file = f.name

    try:
        # Pandas might load it but accessing 'close' will fail
        md = MarketData(temp_file)
        # Accessing close will raise KeyError
        with pytest.raises((KeyError, MarketDataError)):
            md.get(date.fromisoformat("2023-01-02"), "SPX")
    except (MarketDataError, KeyError):
        # If it fails during load, that's also acceptable
        pass
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_compute_state_with_missing_market_data():
    """Test error when computing state but market data is missing."""
    # Create a CSV with missing data for one ticker on a specific date
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date,ticker,close\n")
        f.write("2023-01-02,SPX,4078.447068\n")
        f.write("2023-01-02,SX5E,3771.177591\n")
        f.write("2023-01-02,HSI,21366.71313\n")
        # 2023-01-03 has SPX and SX5E but missing HSI
        f.write("2023-01-03,SPX,4057.98375\n")
        f.write("2023-01-03,SX5E,3754.599846\n")
        # HSI is missing for 2023-01-03
        temp_file = f.name

    try:
        md = MarketData(temp_file)
        strategy = EqualWeightStrategy(
            md=md,
            basket=["SPX", "SX5E", "HSI"],
            seed_date=date.fromisoformat("2023-01-02"),
            calendar=md.get_calendar(),
            initial_index_level=100,
        )

        # Try to compute for a date that exists in calendar but has missing data for HSI
        test_date = date.fromisoformat("2023-01-03")

        # compute_state should raise MarketDataError when trying to get HSI price
        with pytest.raises(MarketDataError, match="No data for 'HSI' on"):
            strategy.compute_state(test_date)
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_get_states_with_invalid_date_range():
    """Test get_states with various invalid date ranges."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    # from_date after to_date - should return empty
    states = get_states(
        strategy, date.fromisoformat("2023-01-10"), date.fromisoformat("2023-01-05")
    )
    assert len(states) == 0


def test_strategy_with_duplicate_tickers():
    """Test strategy with duplicate tickers in basket."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SPX", "SX5E"],  # Duplicate SPX
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    state = strategy.compute_state(strategy.seed_date)

    assert len(state.weights) == 2
    assert state.weights["SPX"] == pytest.approx(1.0 / 3.0, rel=1e-6)  # type: ignore
    assert state.weights["SX5E"] == pytest.approx(1.0 / 3.0, rel=1e-6)  # type: ignore


def test_market_data_update_nonexistent_entry():
    """Test updating market data for non-existent entry (pandas allows this)."""
    md = MarketData("sample_prices.csv")

    # Pandas allows creating new entries via update
    md.update(date.fromisoformat("2020-01-01"), "SPX", 1000.0)

    # Should be able to retrieve it
    price = md.get(date.fromisoformat("2020-01-01"), "SPX")
    assert price == 1000.0


def test_resolve_dates_with_none():
    """Test resolve_dates with None from_date."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    # None should default to seed_date
    schedule = strategy.resolve_dates(None, date.fromisoformat("2023-01-05"))
    assert len(schedule) > 0
    assert strategy.seed_date in schedule


def test_compute_state_after_data_range():
    """Test computing state for date after available data."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )

    # Last date in calendar might fail if it's last day of month (needs next date)
    # Use second-to-last date instead
    dates_list = list(strategy.calendar)
    if len(dates_list) > 1:
        second_last = dates_list[-2]
        state = strategy.compute_state(second_last)
        assert state is not None

    # Date after calendar should fail
    with pytest.raises((ScheduleError, MarketDataError)):
        strategy.compute_state(date.fromisoformat("2024-01-01"))
