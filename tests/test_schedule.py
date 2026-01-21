import pytest
from datetime import date, datetime
from schedule import Schedule, ScheduleError

def test_init_with_string_dates():
    """Test initialization with string dates."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05']
    schedule = Schedule(dates)
    assert len(schedule) == 3
    
def test_init_with_datetime_objects():
    """Test initialization with datetime objects."""
    dates = [datetime(2023, 1, 1), datetime(2023, 1, 3), datetime(2023, 1, 5)]
    schedule = Schedule(dates)
    assert len(schedule) == 3
    
def test_init_with_date_objects():
    """Test initialization with date objects."""
    dates = [date(2023, 1, 1), date(2023, 1, 3), date(2023, 1, 5)]
    schedule = Schedule(dates)
    assert len(schedule) == 3
    
def test_init_removes_duplicates():
    """Test that initialization removes duplicate dates."""
    dates = ['2023-01-01', '2023-01-01', '2023-01-03', '2023-01-03']
    schedule = Schedule(dates)
    assert len(schedule) == 2
    
def test_init_sorts_dates():
    """Test that initialization sorts dates."""
    dates = ['2023-01-05', '2023-01-01', '2023-01-03']
    schedule = Schedule(dates)
    date_list = list(schedule)
    assert date_list == [date(2023, 1, 1), date(2023, 1, 3), date(2023, 1, 5)]
    
def test_prev_with_valid_date():
    """Test prev method with valid target date."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05', '2023-01-10']
    schedule = Schedule(dates)
    
    prev_date = schedule.prev('2023-01-05')
    assert prev_date == date(2023, 1, 3)
    
def test_prev_no_previous_date():
    """Test prev method when no previous date exists."""
    dates = ['2023-01-03', '2023-01-05', '2023-01-10']
    schedule = Schedule(dates)
    
    with pytest.raises(ScheduleError, match="No date before 2023-01-03"):
        schedule.prev('2023-01-03')
        
def test_next_with_valid_date():
    """Test next method with valid target date."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05', '2023-01-10']
    schedule = Schedule(dates)
    
    next_date = schedule.next('2023-01-03')
    assert next_date == date(2023, 1, 5)
    
def test_next_no_following_date():
    """Test next method when no following date exists."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05']
    schedule = Schedule(dates)
    
    with pytest.raises(ScheduleError, match="No date after 2023-01-05"):
        schedule.next('2023-01-05')
        
def test_sub_schedule_inclusive_range():
    """Test sub_schedule with inclusive range."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05', '2023-01-10', '2023-01-15']
    schedule = Schedule(dates)
    
    sub = schedule.sub_schedule('2023-01-03', '2023-01-10')
    sub_dates = list(sub)
    expected = [date(2023, 1, 3), date(2023, 1, 5), date(2023, 1, 10)]
    assert sub_dates == expected
    
def test_sub_schedule_empty_range():
    """Test sub_schedule with range containing no dates."""
    dates = ['2023-01-01', '2023-01-05', '2023-01-10']
    schedule = Schedule(dates)
    
    sub = schedule.sub_schedule('2023-02-01', '2023-02-28')
    assert len(sub) == 0
    
def test_sub_schedule_with_date_objects():
    """Test sub_schedule with date objects."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05', '2023-01-10']
    schedule = Schedule(dates)
    
    sub = schedule.sub_schedule(date(2023, 1, 2), date(2023, 1, 6))
    sub_dates = list(sub)
    expected = [date(2023, 1, 3), date(2023, 1, 5)]
    assert sub_dates == expected
    
def test_is_last_day_of_month_true():
    """Test is_last_day_of_month when date is last day of month."""
    dates = ['2023-01-30', '2023-01-31', '2023-02-01', '2023-02-02']
    schedule = Schedule(dates)
    
    assert schedule.is_last_day_of_month(date(2023, 1, 31)) == True
    
def test_is_last_day_of_month_false():
    """Test is_last_day_of_month when date is not last day of month."""
    dates = ['2023-01-30', '2023-01-31', '2023-02-01', '2023-02-02']
    schedule = Schedule(dates)
    
    assert schedule.is_last_day_of_month(date(2023, 2, 1)) == False
    
def test_is_last_day_of_month_no_next_date():
    """Test is_last_day_of_month when no next date exists."""
    dates = ['2023-01-30', '2023-01-31']
    schedule = Schedule(dates)
    
    with pytest.raises(ScheduleError):
        schedule.is_last_day_of_month(date(2023, 1, 31))
        
def test_iteration():
    """Test that Schedule is iterable and yields date objects."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05']
    schedule = Schedule(dates)
    
    date_list = list(schedule)
    expected = [date(2023, 1, 1), date(2023, 1, 3), date(2023, 1, 5)]
    assert date_list == expected
    
    # Test that iteration yields date objects, not Timestamps
    for dt in schedule:
        assert isinstance(dt, date)
        
def test_len():
    """Test __len__ method."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05']
    schedule = Schedule(dates)
    assert len(schedule) == 3
    
    empty_schedule = Schedule([])
    assert len(empty_schedule) == 0
    
def test_repr():
    """Test __repr__ method."""
    dates = ['2023-01-01', '2023-01-03', '2023-01-05']
    schedule = Schedule(dates)
    
    repr_str = repr(schedule)
    assert "Schedule(3 dates:" in repr_str
    assert "2023-01-01 to 2023-01-05" in repr_str
    
def test_empty_schedule():
    """Test behavior with empty schedule."""
    schedule = Schedule([])
    assert len(schedule) == 0
    
    with pytest.raises(ScheduleError):
        schedule.prev('2023-01-01')
        
    with pytest.raises(ScheduleError):
        schedule.next('2023-01-01')
        
def test_single_date_schedule():
    """Test behavior with single date schedule."""
    schedule = Schedule(['2023-01-01'])
    assert len(schedule) == 1
    
    with pytest.raises(ScheduleError):
        schedule.prev('2023-01-01')
        
    with pytest.raises(ScheduleError):
        schedule.next('2023-01-01')
        
    with pytest.raises(ScheduleError):
        schedule.is_last_day_of_month(date(2023, 1, 1))

def test_mixed_date_formats():
    """Test initialization with mixed date formats."""
    dates = ['2023-01-01', datetime(2023, 1, 3), date(2023, 1, 5)]
    schedule = Schedule(dates)
    
    date_list = list(schedule)
    expected = [date(2023, 1, 1), date(2023, 1, 3), date(2023, 1, 5)]
    assert date_list == expected

def test_month_boundary_scenarios():
    """Test month boundary scenarios for is_last_day_of_month."""
    # Test February to March transition
    dates = ['2023-02-27', '2023-02-28', '2023-03-01', '2023-03-02']
    schedule = Schedule(dates)
    
    assert schedule.is_last_day_of_month(date(2023, 2, 28)) == True
    assert schedule.is_last_day_of_month(date(2023, 2, 27)) == False
    
    # Test leap year February
    dates_leap = ['2024-02-28', '2024-02-29', '2024-03-01']
    schedule_leap = Schedule(dates_leap)
    
    assert schedule_leap.is_last_day_of_month(date(2024, 2, 29)) == True
    assert schedule_leap.is_last_day_of_month(date(2024, 2, 28)) == False

