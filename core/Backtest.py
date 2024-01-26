import numpy as np
import numba as nb
from numba import f8, i8

# Define the data type for trades
trade_t = np.dtype([
    ('is_buy', 'b1'),
    ('risk', 'f8'),
    ('sl', 'f8'),
    ('tp', 'f8'),
    ('open_idx', 'f8'),
    ('close_idx', 'f8'),
    ('profit', 'f8')
])

# Numba equivalent of the trade_t data type
tr = nb.from_dtype(trade_t)

# Define a default trade with np.nan values
default_trade = np.array([(False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)], dtype=trade_t)


# Specify the data types for the elements in the class
default_spec = {
    'op': f8[:],
    'high': f8[:],
    'low': f8[:],
    'close': f8[:],
    'trades': tr[:],
    'fill': i8,
    'active': i8[:],
    'index': i8,
    'trade_halt': i8,
    'entry_price': f8,
    'winning_trades': i8,
    'completed_trades': i8,
    'returns': f8,
    'max_profit': f8,
    'max_drawdown': f8,
    'return_y': f8[:],
    'last_trade_idx': i8

}

