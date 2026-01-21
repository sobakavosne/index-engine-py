"""
Tests for MarketData class.
"""

import pytest
import tempfile
import os
from datetime import date
from marketdata import MarketData, MarketDataError


def test_load_valid_csv():
    """Test loading a valid CSV file."""
    md = MarketData("sample_prices.csv")
    assert md is not None
    price = md.get(date.fromisoformat("2023-01-02"), "SPX")
    assert price > 0


def test_load_nonexistent_file():
    """Test loading a file that doesn't exist."""
    with pytest.raises(MarketDataError, match="File not found"):
        MarketData("nonexistent_file.csv")


def test_load_invalid_csv():
    """Test loading an invalid CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("invalid,data\n")
        f.write("not,proper,format\n")
        temp_file = f.name

    try:
        with pytest.raises(MarketDataError):
            MarketData(temp_file)
    finally:
        os.unlink(temp_file)


def test_get_valid_price():
    """Test getting a valid price for a date and ticker."""
    md = MarketData("sample_prices.csv")
    price = md.get(date.fromisoformat("2023-01-02"), "SPX")
    assert price == pytest.approx(4078.447068, rel=1e-6)  # type: ignore


def test_get_different_tickers():
    """Test getting prices for different tickers."""
    md = MarketData("sample_prices.csv")
    spx_price = md.get(date.fromisoformat("2023-01-02"), "SPX")
    sx5e_price = md.get(date.fromisoformat("2023-01-02"), "SX5E")
    hsi_price = md.get(date.fromisoformat("2023-01-02"), "HSI")

    assert spx_price > 0
    assert sx5e_price > 0
    assert hsi_price > 0
    assert spx_price != sx5e_price
    assert sx5e_price != hsi_price


def test_get_invalid_date():
    """Test getting price for a date that doesn't exist."""
    md = MarketData("sample_prices.csv")
    with pytest.raises(MarketDataError, match="No data for 'SPX' on"):
        md.get(date.fromisoformat("2020-01-01"), "SPX")


def test_get_invalid_ticker():
    """Test getting price for a ticker that doesn't exist."""
    md = MarketData("sample_prices.csv")
    with pytest.raises(MarketDataError, match="No data for 'INVALID' on"):
        md.get(date.fromisoformat("2023-01-02"), "INVALID")


def test_get_calendar():
    """Test getting the calendar schedule."""
    md = MarketData("sample_prices.csv")
    calendar = md.get_calendar()

    assert calendar is not None
    assert len(calendar) > 0
    dates = list(calendar)
    assert dates == sorted(dates)
    assert dates[0] == date.fromisoformat("2023-01-02")
    assert dates[-1] == date.fromisoformat("2023-06-30")


def test_update_price():
    """Test updating a price in memory."""
    md = MarketData("sample_prices.csv")
    original_price = md.get(date.fromisoformat("2023-01-02"), "SPX")

    new_price = 5000.0
    md.update(date.fromisoformat("2023-01-02"), "SPX", new_price)

    updated_price = md.get(date.fromisoformat("2023-01-02"), "SPX")
    assert updated_price == new_price
    assert updated_price != original_price


def test_update_multiple_prices():
    """Test updating multiple prices."""
    md = MarketData("sample_prices.csv")

    md.update(date.fromisoformat("2023-01-02"), "SPX", 5000.0)
    md.update(date.fromisoformat("2023-01-02"), "SX5E", 6000.0)
    md.update(date.fromisoformat("2023-01-03"), "HSI", 7000.0)

    assert md.get(date.fromisoformat("2023-01-02"), "SPX") == 5000.0
    assert md.get(date.fromisoformat("2023-01-02"), "SX5E") == 6000.0
    assert md.get(date.fromisoformat("2023-01-03"), "HSI") == 7000.0


def test_update_invalid_date():
    """Test updating a price for a date that doesn't exist (pandas allows this)."""
    md = MarketData("sample_prices.csv")
    md.update(date.fromisoformat("2020-01-01"), "SPX", 1000.0)
    assert md.get(date.fromisoformat("2020-01-01"), "SPX") == 1000.0


def test_update_invalid_ticker():
    """Test updating a price for a ticker that doesn't exist (pandas allows this)."""
    md = MarketData("sample_prices.csv")
    md.update(date.fromisoformat("2023-01-02"), "INVALID", 1000.0)
    assert md.get(date.fromisoformat("2023-01-02"), "INVALID") == 1000.0


def test_get_updated_dates():
    """Test tracking of updated dates."""
    md = MarketData("sample_prices.csv")

    updated = md.get_updated_dates()
    assert len(updated) == 0

    md.update(date.fromisoformat("2023-01-02"), "SPX", 5000.0)
    updated = md.get_updated_dates()
    assert date.fromisoformat("2023-01-02") in updated
    assert len(updated) == 1

    md.update(date.fromisoformat("2023-01-03"), "SX5E", 6000.0)
    updated = md.get_updated_dates()
    assert date.fromisoformat("2023-01-02") in updated
    assert date.fromisoformat("2023-01-03") in updated
    assert len(updated) == 2


def test_clear_updated_dates():
    """Test clearing the updated dates tracking."""
    md = MarketData("sample_prices.csv")

    md.update(date.fromisoformat("2023-01-02"), "SPX", 5000.0)
    assert len(md.get_updated_dates()) == 1

    md.clear_updated_dates()
    assert len(md.get_updated_dates()) == 0

    assert md.get(date.fromisoformat("2023-01-02"), "SPX") == 5000.0


def test_register_update_callback():
    """Test registering and calling update callbacks."""
    md = MarketData("sample_prices.csv")

    callback_calls = []

    def callback(updated_date: date):
        callback_calls.append(updated_date) # type: ignore

    md.register_update_callback(callback)

    md.update(date.fromisoformat("2023-01-02"), "SPX", 5000.0)
    assert len(callback_calls) == 1 # type: ignore
    assert callback_calls[0] == date.fromisoformat("2023-01-02")

    md.update(date.fromisoformat("2023-01-03"), "SX5E", 6000.0)
    assert len(callback_calls) == 2 # type: ignore
    assert callback_calls[1] == date.fromisoformat("2023-01-03")


def test_multiple_update_callbacks():
    """Test registering multiple callbacks."""
    md = MarketData("sample_prices.csv")

    calls1 = []
    calls2 = []

    def callback1(updated_date: date):
        calls1.append(updated_date) # type: ignore

    def callback2(updated_date: date):
        calls2.append(updated_date) # type: ignore

    md.register_update_callback(callback1)
    md.register_update_callback(callback2)

    md.update(date.fromisoformat("2023-01-02"), "SPX", 5000.0)

    assert len(calls1) == 1 # type: ignore
    assert len(calls2) == 1 # type: ignore
    assert calls1[0] == date.fromisoformat("2023-01-02")
    assert calls2[0] == date.fromisoformat("2023-01-02")


def test_empty_csv():
    """Test handling of empty CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date,ticker,close\n")
        temp_file = f.name

    try:
        md = MarketData(temp_file)
        calendar = md.get_calendar()
        assert len(calendar) == 0
    finally:
        os.unlink(temp_file)


def test_single_row_csv():
    """Test handling of CSV with single row of data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date,ticker,close\n")
        f.write("2023-01-02,SPX,1000.0\n")
        temp_file = f.name

    try:
        md = MarketData(temp_file)
        price = md.get(date.fromisoformat("2023-01-02"), "SPX")
        assert price == 1000.0
        calendar = md.get_calendar()
        assert len(calendar) == 1
    finally:
        os.unlink(temp_file)
