"""
Integration tests for end-to-end workflow.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
from datetime import date

from marketdata import MarketData
from rule import EqualWeightStrategy
from runner import get_states


def test_main_produces_expected_output():
    """Test that running main.py produces output matching expected_output.csv."""
    project_root = Path(__file__).parent.parent
    main_script = project_root / "main.py"
    expected_output = project_root / "expected_output.csv"
    sample_output = project_root / "sample_output.csv"

    if sample_output.exists():
        sample_output.unlink()

    try:
        result = subprocess.run(
            [sys.executable, str(main_script)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"main.py failed: {result.stderr}"

        assert sample_output.exists(), "sample_output.csv was not created"

        expected_df = pd.read_csv(expected_output)  # type: ignore
        sample_df = pd.read_csv(sample_output)  # type: ignore

        assert list(expected_df.columns) == list(sample_df.columns), "Column names don't match"
        assert len(expected_df) == len(sample_df), "Row counts don't match"

        expected_df["date"] = pd.to_datetime(expected_df["date"]).dt.date
        sample_df["date"] = pd.to_datetime(sample_df["date"]).dt.date
        assert list(expected_df["date"]) == list(sample_df["date"]), "Dates don't match"

        for idx, (expected_row, sample_row) in enumerate(zip(expected_df.itertuples(), sample_df.itertuples())):
            expected_level = float(expected_row.index_level)  # type: ignore
            sample_level = float(sample_row.index_level)  # type: ignore
            assert abs(expected_level - sample_level) < 1e-6, (
                f"Index level mismatch at row {idx + 1} (date {expected_row.date}): "
                f"expected {expected_level}, got {sample_level}"
            )
    finally:
        if sample_output.exists():
            sample_output.unlink()


def test_get_states_matches_main_output():
    """Test that get_states produces the same results as main.py would."""
    md = MarketData("sample_prices.csv")
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )
    states = get_states(strategy, None, date.fromisoformat("2023-06-29"))

    expected_output = Path(__file__).parent.parent / "expected_output.csv"
    expected_df = pd.read_csv(expected_output)  # type: ignore
    expected_df["date"] = pd.to_datetime(expected_df["date"]).dt.date

    computed_df = pd.DataFrame(
        [{"date": date_key, "index_level": state.index_level} for date_key, state in states.items()]
    )

    assert len(computed_df) == len(expected_df), "Row counts don't match"

    assert list(computed_df["date"]) == list(expected_df["date"]), "Dates don't match"

    for idx, (expected_row, computed_row) in enumerate(zip(expected_df.itertuples(), computed_df.itertuples())):
        expected_level = float(expected_row.index_level)  # type: ignore
        computed_level = float(computed_row.index_level)  # type: ignore
        assert abs(expected_level - computed_level) < 1e-6, (
            f"Index level mismatch at row {idx + 1} (date {expected_row.date}): "
            f"expected {expected_level}, got {computed_level}"
        )


def test_main_output_format():
    """Test that main.py produces correctly formatted CSV output."""
    project_root = Path(__file__).parent.parent
    main_script = project_root / "main.py"
    sample_output = project_root / "sample_output.csv"

    if sample_output.exists():
        sample_output.unlink()

    try:
        result = subprocess.run(
            [sys.executable, str(main_script)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"main.py failed: {result.stderr}"
        assert sample_output.exists(), "sample_output.csv was not created"

        df = pd.read_csv(sample_output)  # type: ignore

        assert list(df.columns) == ["date", "index_level"], "Incorrect column names"

        df["date"] = pd.to_datetime(df["date"])
        assert df["date"].notna().all(), "Some dates are invalid"

        assert pd.api.types.is_numeric_dtype(df["index_level"]), "index_level is not numeric"

        assert df["index_level"].notna().all(), "Some index_level values are missing"

        assert (df["index_level"] > 0).all(), "Some index_level values are non-positive"

        assert df["date"].is_monotonic_increasing, "Dates are not in chronological order"
    finally:
        if sample_output.exists():
            sample_output.unlink()


def test_main_output_completeness():
    """Test that main.py produces output for all expected dates."""
    project_root = Path(__file__).parent.parent
    main_script = project_root / "main.py"
    sample_output = project_root / "sample_output.csv"
    expected_output = project_root / "expected_output.csv"

    if sample_output.exists():
        sample_output.unlink()

    try:
        result = subprocess.run(
            [sys.executable, str(main_script)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert sample_output.exists()

        sample_df = pd.read_csv(sample_output)  # type: ignore
        expected_df = pd.read_csv(expected_output)  # type: ignore

        sample_df["date"] = pd.to_datetime(sample_df["date"]).dt.date
        expected_df["date"] = pd.to_datetime(expected_df["date"]).dt.date

        assert len(sample_df) == len(expected_df), (
            f"Row count mismatch: expected {len(expected_df)}, got {len(sample_df)}"
        )

        expected_dates = set(expected_df["date"])
        sample_dates = set(sample_df["date"])
        missing_dates = expected_dates - sample_dates
        extra_dates = sample_dates - expected_dates

        assert not missing_dates, f"Missing dates in output: {missing_dates}"
        assert not extra_dates, f"Extra dates in output: {extra_dates}"
    finally:
        if sample_output.exists():
            sample_output.unlink()

