import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

import pytz

if not mt5.initialize():
    print("Could not initialize: ", mt5.last_error())
    mt5.shutdown()

pairs = ['EURUSD']
timeframes = [mt5.TIMEFRAME_M5]
bars = 99999

timeframe_dict = {mt5.TIMEFRAME_M1: '1m',
                  mt5.TIMEFRAME_M2: '2m',
                  mt5.TIMEFRAME_M3: '3m',
                  mt5.TIMEFRAME_M4: '4m',
                  mt5.TIMEFRAME_M5: '5m',
                  mt5.TIMEFRAME_M6: '6m',
                  mt5.TIMEFRAME_M10: '10m',
                  mt5.TIMEFRAME_M12: '12m',
                  mt5.TIMEFRAME_M15: '15m',
                  mt5.TIMEFRAME_M20: '20m',
                  mt5.TIMEFRAME_M30: '30m',
                  mt5.TIMEFRAME_H1: '1h',
                  mt5.TIMEFRAME_H2: '2h',
                  mt5.TIMEFRAME_H3: '3h',
                  mt5.TIMEFRAME_H4: '4h',
                  mt5.TIMEFRAME_H6: '6h',
                  mt5.TIMEFRAME_H8: '8h',
                  mt5.TIMEFRAME_H12: '12h',
                  mt5.TIMEFRAME_D1: '1d',
                  mt5.TIMEFRAME_W1: '1w',
                  mt5.TIMEFRAME_MN1: '1mo'}

# set time zone to UTC
timezone = pytz.timezone("Etc/UTC")

# create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset

utc_from = datetime.now(timezone)

for pair in pairs:
    for t in timeframes:
        rates = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M5, utc_from, bars)

        df = pd.DataFrame(rates)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        df.to_csv(f'data/{pair}_{timeframe_dict[t]}_{utc_from.date()}_{bars // 1000}k.csv', index=False)

# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()

print('Data saved to csv file')
