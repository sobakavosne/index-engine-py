import pandas as pd
from datetime import date, datetime
from typing import Any, Union, Iterator

class ScheduleError(Exception):
    """Custom exception for Schedule errors"""
    pass

class Schedule:
    """
    A class that wraps a DatetimeIndex for efficient date operations.
    """
    
    def __init__(self, data: Any): # Pandas types don't seem to be properly exposed
        """
        Initialize Schedule with date data.
        
        Args:
            data: ArrayLike containing dates (strings, datetime objects, etc.)
        """
        self._index = pd.DatetimeIndex(data).sort_values().drop_duplicates()
    
    def prev(self, target_date: Union[date, datetime, str]) -> date:
        """
        Get the previous date before the given date.
        
        Args:
            target_date: The reference date
            
        Returns:
            date: The previous date in the schedule
            
        Raises:
            ScheduleError: If no previous date exists
        """
        target_ts = pd.Timestamp(target_date)
        previous_dates = self._index[self._index < target_ts]
        
        if len(previous_dates) == 0:
            raise ScheduleError(f"No date before {target_date} in schedule")
        
        return previous_dates.max().date()
    
    def next(self, target_date: Union[date, datetime, str]) -> date:
        """
        Get the next date after the given date.
        
        Args:
            target_date: The reference date
            
        Returns:
            date: The next date in the schedule
            
        Raises:
            ScheduleError: If no next date exists
        """
        target_ts = pd.Timestamp(target_date)
        following_dates = self._index[self._index > target_ts]
        
        if len(following_dates) == 0:
            raise ScheduleError(f"No date after {target_date} in schedule")
        
        return following_dates.min().date()

    def sub_schedule(
            self,
            start_date: Union[date, datetime, str], 
            end_date: Union[date, datetime, str],
        ) -> 'Schedule':
        """
        Create a new Schedule with dates within the given range (inclusive).
        
        Args:
            start_date: Start of the range (inclusive)
            end_date: End of the range (inclusive)
            
        Returns:
            Schedule: New Schedule containing dates in the range
        """
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        subset = self._index[(self._index >= start_ts) & (self._index <= end_ts)]
        return Schedule(subset)
    
    def is_last_day_of_month(self, target_date: date) -> bool:
        """Return true if target_date is the last day of the month in this schedule.

        Args:
            target_date: The reference date
            
        Returns:
            date: True if target_date is the last day of the month in this schedule
            
        Raises:
            ScheduleError: If no next date exists
        """
        next_date = self.next(target_date)
        return target_date.month != next_date.month
    
    def __iter__(self) -> Iterator[date]:
        """Make Schedule enumerable, yielding date objects."""
        for ts in self._index:
            yield ts.date()
    
    def __len__(self) -> int:
        """Return the number of dates in the schedule."""
        return len(self._index)
    
    def __repr__(self) -> str:
        """Return string representation of Schedule."""
        return f"Schedule({len(self)} dates: {self._index.min().date()} to {self._index.max().date()})"
