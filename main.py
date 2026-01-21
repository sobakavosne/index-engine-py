from datetime import date

import pandas as pd
from marketdata import MarketData
from rule import EqualWeightStrategy
from runner import get_states
from lock_manager import ThreadingLockManager

if __name__ == "__main__":
    # Create lock manager for thread-safe operations
    lock_manager = ThreadingLockManager()
    
    # Create MarketData (thread-safe with internal lock)
    md = MarketData('sample_prices.csv')
    
    # Create strategy
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )
    
    # Set lock manager for thread-safe computation
    strategy.set_lock_manager(lock_manager)
    
    # Compute states (now thread-safe)
    states = get_states(strategy, None, date.fromisoformat("2023-06-29"))
    df = pd.DataFrame([
        {'date': date_key, 'index_level': state.index_level}
        for date_key, state in states.items()
    ])
    df.to_csv('sample_output.csv', index=False)