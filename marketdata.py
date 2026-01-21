from threading import RLock
import pandas as pd

from typing import cast, Set, List, Callable
from datetime import date

from schedule import Schedule

class MarketDataError(Exception):
    """Custom exception for MarketData errors"""
    pass

class MarketData:
    """
    A class to load and query market data from a CSV file.
    
    The CSV file should have columns: date, ticker, close
    
    Thread Safety:
    - All operations are thread-safe using an internal RLock
    - DataFrame operations are protected by locks
    - Callback invocations are thread-safe
    """
    
    def __init__(self, filename: str):
        """
        Initialize MarketData with a CSV file.
        
        Args:
            filename (str): Path to the CSV file containing market data
        """
        self._data = self._load_data(filename)
        # Track which dates have been updated for cache invalidation
        self._updated_dates: Set[date] = set()
        # Callbacks to notify when data is updated
        self._update_callbacks: List[Callable[[date], None]] = []
        # Internal lock for thread-safe operations
        self._internal_lock = RLock()

    def _load_data(self, filename: str) -> pd.DataFrame:
        """Load from a CSV file."""
        try:
            df = pd.read_csv(filename)  # type: ignore

            # Convert date column to datetime
            df["date"] = pd.to_datetime(df["date"])

            # Set multi-index for fast lookups
            df = df.set_index(["date", "ticker"])

            return df

        except FileNotFoundError:
            raise MarketDataError(f"File not found: {filename}")
        except Exception as e:
            raise MarketDataError(f"Error loading data from {filename}: {e}")

    def get(self, date: date, ticker: str) -> float:
        """
        Get the closing price for a specific date and ticker.

        Thread-safe: Uses internal lock to protect DataFrame access.

        Args:
            date: Date to query
            ticker: Ticker symbol (e.g., 'SPX', 'SX5E', 'HSI')

        Returns:
            float: The closing price for the given date and ticker

        Raises:
            MarketDataError: If the requested date/ticker combination is not found
        """
        with self._internal_lock:
            try:
                return cast(float, self._data.loc[(pd.to_datetime(date), ticker), "close"])
            except KeyError:
                raise MarketDataError(f"No data for '{ticker}' on {date}.")

    def get_calendar(self) -> Schedule:
        """
        Get all available dates in the dataset.

        Returns:
            Schedule: Sorted list of all unique dates in the dataset
        """
        return Schedule(self._data.index.get_level_values("date"))

    def update(self, date: date, ticker: str, price: float):
        """
        Update a price in memory.

        Thread-safe: Uses internal lock to protect DataFrame and callback operations.

        Args:
            date: The date of the price to update
            ticker: The ticker symbol
            price: The new price value

        Raises:
            MarketDataError: If the date/ticker combination doesn't exist
        """
        date_ts = pd.to_datetime(date)
        # Copy callbacks list to avoid modification during iteration
        callbacks_copy = []
        with self._internal_lock:
            try:
                self._data.loc[(date_ts, ticker), "close"] = price
                # Track that this date has been updated for cache invalidation
                self._updated_dates.add(date)
                # Copy callbacks to avoid holding lock during callback execution
                callbacks_copy = list(self._update_callbacks)
            except KeyError:
                raise MarketDataError(f"No data for '{ticker}' on {date}.")
        
        # Notify callbacks outside of lock to avoid deadlocks
        # (callbacks may acquire other locks)
        for callback in callbacks_copy:
            callback(date)

    def register_update_callback(self, callback: Callable[[date], None]):
        """
        Register a callback to be called when market data is updated.

        Thread-safe: Uses internal lock.

        Args:
            callback: A function that takes a date parameter
        """
        with self._internal_lock:
            self._update_callbacks.append(callback)

    def get_updated_dates(self) -> Set[date]:
        """
        Get the set of dates that have been updated.

        Thread-safe: Uses internal lock.

        Returns:
            Set of dates that have been modified via update()
        """
        with self._internal_lock:
            return self._updated_dates.copy()

    def clear_updated_dates(self):
        """
        Clear the tracking of updated dates.
        
        Thread-safe: Uses internal lock.
        """
        with self._internal_lock:
            self._updated_dates.clear()
