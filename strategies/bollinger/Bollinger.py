import numba as nb
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Quad
from bokeh.plotting import figure
from numba import njit, void, i8, f8, b1, jit, types, objmode
from numba.core.types import Tuple, UniTuple
from numba.experimental import jitclass
from numpy import ndarray
from numba.typed import Dict
from core import Indicators as I
from core.Backtest import default_trade, trade_t, default_spec

spec = default_spec.copy()
spec['lower'] = f8[:]
spec['sma'] = f8[:]
spec['upper'] = f8[:]


@jitclass(spec)
class Strategy:

    def __init__(self, op: ndarray, high: ndarray, low: ndarray, close: ndarray):
        # Initialize bar data
        self.op = op
        self.high = high
        self.low = low
        self.close = close

        # Initialize trade data
        self.trades = np.full(100, default_trade[0], dtype=trade_t)
        self.fill = 0
        self.active = np.full(1, -1, dtype=np.int64)
        self.index = 0
        self.trade_halt = 0

        # Initialize statistics
        self.winning_trades = 0
        self.completed_trades = 0
        self.returns = 0
        self.max_profit = 0
        self.max_drawdown = 0
        self.return_y = np.zeros(len(self.op))
        self.last_trade_idx = 0

        # Initialize indicators
        self.lower, self.sma, self.upper = I.bollinger_bands(self.close, 20)

    def run(self):
        """
        Run the backtest
        """

        # Loop through each candle
        for i in range(len(self.close) - 1):
            self.index = i
            self.entry_price = self.op[i + 1]

            self.check_trades()
            if self.trade_halt == 0:
                self.next()
            else:
                self.trade_halt -= 1

        stats, trades, return_y = self.calculate_stats()

        self.clear_data()

        return stats, trades, return_y

    def get_data(self, data, offset=0):
        """
        Get the data for the current candle.

        :param data: A numpy array with data for each bar
        :param offset: How many bars to look back
        :return: Data for the current candle - offset
        """

        if offset > self.index:
            return np.nan

        return data[self.index - offset]

    def add_trade(self, is_buy, risk, sl, tp):
        """
        Add a trade to the trade list.

        :param is_buy: Whether the trade is a buy or sell
        :param risk: How much capital to risk in percentage
        :param sl: Price of the stop loss
        :param tp: Price of the take profit
        :param open_idx: Index of the candle when the trade was opened
        :return: Index of the trade
        """

        if sl == self.entry_price or tp == self.entry_price:
            return -1

        # Add more space to the trade list if it is full
        if self.fill == len(self.trades):
            new_trades = np.full(self.fill * 2, default_trade[0], dtype=trade_t)
            new_trades[:self.fill] = self.trades
            self.trades = new_trades

        # Add the trade to the active list
        for i in range(len(self.active)):
            if self.active[i] == -1:
                self.active[i] = self.fill
                break

            if i == len(self.active) - 1:
                return -1

        # Fill in the trade details
        self.trades[self.fill]['is_buy'] = is_buy
        self.trades[self.fill]['risk'] = risk
        self.trades[self.fill]['sl'] = sl
        self.trades[self.fill]['tp'] = tp
        self.trades[self.fill]['open_idx'] = self.index + 1

        self.fill += 1

        return self.fill - 1

    def buy(self, risk, sl=np.nan, tp=np.nan, rrr=np.nan):
        """
        Open a long position.

        :param risk: How much capital to risk in percentage
        :param sl: Price of the stop loss
        :param rrr: How much to risk relative to the stop loss
        :param tp: Price of the take profit. If not specified, it will be calculated from the rrr.
        :return: Index of the trade
        """

        if rrr == 0:
            raise ValueError('rrr cannot be 0')

        # Check if one of sl or tp is nan if rrr is also nan
        if np.isnan(sl) and not (np.isnan(rrr) and np.isnan(tp)):
            raise ValueError('Either rrr and tp, or sl must be specified')

        if np.isnan(tp) and not np.isnan(rrr):
            tp = self.entry_price + rrr * (self.entry_price - sl)

        if np.isnan(sl):
            sl = self.entry_price - (1 / rrr) * (tp - self.entry_price)

        if not np.isnan(tp) and tp < self.entry_price:
            return -1

        return self.add_trade(True, risk, sl,
                              tp if not np.isnan(tp) else self.entry_price + rrr * (self.entry_price - sl))

    def sell(self, risk, sl=np.nan, tp=np.nan, rrr=np.nan):
        """
        Open a short position.

        :param risk: How much capital to risk in percentage
        :param sl: Price of the stop loss
        :param rrr: How much to risk relative to the stop loss
        :param tp: Price of the take profit. If not specified, it will be calculated from the rrr.
        :return: Index of the trade
        """

        if rrr == 0:
            raise ValueError('rrr cannot be 0')

        # Check if either rrr or tp is specified
        if np.isnan(sl) and not (np.isnan(rrr) and np.isnan(tp)):
            raise ValueError('Either sl, or tp and rrr must be specified')

        if np.isnan(tp) and not np.isnan(rrr):
            tp = self.entry_price - rrr * (sl - self.entry_price)

        if np.isnan(sl):
            sl = self.entry_price + (1 / rrr) * (self.entry_price - tp)

        if tp > self.entry_price:
            return -1

        return self.add_trade(False, risk, sl,
                              tp if not np.isnan(tp) else self.entry_price - rrr * (sl - self.entry_price))

    def check_trades(self):
        """
        Check if any active trades have reached their stop loss or take profit and close them.
        """

        def update_stats(profit):
            """
            Update the statistics of the backtest according to the new trade.
            :param profit: The profit of the newly completed trade
            """

            self.completed_trades += 1
            self.winning_trades += 1 if profit > 0 else 0
            self.return_y[self.last_trade_idx:self.index] = self.returns
            self.last_trade_idx = self.index
            self.returns += profit
            self.max_profit = max(self.max_profit, self.returns)
            self.max_drawdown = min(self.max_drawdown, self.returns - self.max_profit)

        for i in range(len(self.active)):
            if self.active[i] == -1:
                continue

            # Get the trade
            trade = self.trades[self.active[i]]

            # Check if the trade is a buy or sell
            if trade['is_buy']:

                # If it hit the stop loss
                if self.low[self.index] <= trade['sl']:
                    trade['close_idx'] = self.index
                    trade['profit'] = -trade['risk']
                    self.active[i] = -1

                    # Update statistics
                    update_stats(trade['profit'])

                # Check if the trade has reached its take profit
                elif self.high[self.index] >= trade['tp']:
                    trade['close_idx'] = self.index
                    trade['profit'] = trade['risk'] * ((trade['tp'] - self.op[int(trade['open_idx'])]) /
                                                       (self.op[int(trade['open_idx'])] - trade['sl']))
                    self.active[i] = -1

                    # Update statistics
                    update_stats(trade['profit'])
            else:
                # If it hit the stop loss
                if self.high[self.index] >= trade['sl']:
                    trade['close_idx'] = self.index
                    trade['profit'] = -trade['risk']
                    self.active[i] = -1

                    # Update statistics
                    update_stats(trade['profit'])

                # If it hit the take profit
                elif self.low[self.index] <= trade['tp']:
                    trade['close_idx'] = self.index
                    trade['profit'] = trade['risk'] * ((self.op[int(trade['open_idx'])] - trade['tp']) /
                                                       (trade['sl'] - self.op[int(trade['open_idx'])]))
                    self.active[i] = -1

                    # Update statistics
                    update_stats(trade['profit'])

    def calculate_stats(self):
        """
        Calculate the statistics of the finished backtest
        """

        self.return_y[self.last_trade_idx:] = self.returns
        win_rate = (self.winning_trades / self.completed_trades) if self.completed_trades != 0 else 0

        stats = Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.float64)
        stats['win_rate'] = win_rate
        stats['returns'] = self.returns
        stats['max_profit'] = self.max_profit
        stats['max_drawdown'] = self.max_drawdown

        return stats, self.trades, self.return_y

    def next(self):
        """
        Run the strategy for the next candle
        """

        break_below_red = self.get_data(self.close, 1) < self.get_data(self.lower, 1) and self.get_data(self.close) > \
                          self.get_data(self.lower)

        green_rebound = break_below_red and self.get_data(self.op) > self.get_data(self.close)

        if green_rebound:
            self.buy(1, sl=self.get_data(self.low), rrr=1)

    def clear_data(self):
        """
        Clear the data from the previous backtest
        """

        self.trades = np.full(100, default_trade[0], dtype=trade_t)
        self.fill = 100
        self.active = np.full(1, -1, dtype=np.int64)
        self.index = 0
        self.trade_halt = 200
        self.entry_price = 3

        self.winning_trades = 0
        self.completed_trades = 0
        self.returns = 0
        self.max_profit = 0
        self.max_drawdown = 0
        self.return_y = np.zeros(len(self.op))
        self.last_trade_idx = 0


def plot(op, high, low, close, trades, lower, sma, upper, return_y):
    """
    Plot the backtest results.
    """

    # Assuming 'returns', 'op', 'high', 'low', 'close', 'ema_short', and 'sma' are your data arrays

    # Create a ColumnDataSource with your data
    source = ColumnDataSource(data=dict(
        index=list(range(len(op))),
        open=op,
        high=high,
        low=low,
        close=close,
        lower=lower,
        sma=sma,
        upper=upper,
        returns=return_y
    ))

    # Create a new figure for returns
    p1 = figure(width=800, height=250, title="Returns")
    p1.line('index', 'returns', source=source, color='green')

    # Create a new figure for OHLC
    p2 = figure(width=800, height=250, x_range=p1.x_range, title="OHLC Prices with SMA", aspect_ratio=3)
    p2.segment('index', 'high', 'index', 'low', color="black", source=source)

    source.data['colour'] = ['green' if c > o else 'red' for c, o in
                             zip(source.data['close'], source.data['open'])]
    p2.vbar('index', 0.5, 'open', 'close', fill_color='colour', line_color="black", source=source)

    # Assuming 'trades' is a list of your trades
    for t in trades:

        if np.isnan(t['close_idx']):
            continue

        if t['is_buy']:
            # Create a Quad glyph for the trade
            sl_glyph = Quad(top=op[int(t['open_idx'])], bottom=t['sl'], left=t['open_idx'], right=t['close_idx'],
                            fill_color='red', fill_alpha=0.5)

            tp_glyph = Quad(top=t['tp'], bottom=op[int(t['open_idx'])], left=t['open_idx'], right=t['close_idx'],
                            fill_color='green', fill_alpha=0.5)
        else:
            sl_glyph = Quad(top=t['sl'], bottom=op[int(t['open_idx'])], left=t['open_idx'], right=t['close_idx'],
                            fill_color='red', fill_alpha=0.5)

            tp_glyph = Quad(top=op[int(t['open_idx'])], bottom=t['tp'], left=t['open_idx'], right=t['close_idx'],
                            fill_color='green', fill_alpha=0.5)

        p2.add_glyph(sl_glyph)
        p2.add_glyph(tp_glyph)

    p2.line('index', 'sma', source=source, color='blue', legend_label='SMA')
    p2.varea('index', 'lower', 'upper', source=source, fill_alpha=0.2, fill_color='blue',
             legend_label='Bollinger Bands')

    # Output to static HTML file
    output_file("plot.html")

    # Show the plot
    show(column(p1, p2))


def backtest(data):
    op_prices = data['Open'].to_numpy()
    high_prices = data['High'].to_numpy()
    low_prices = data['Low'].to_numpy()
    close_prices = data['Close'].to_numpy()

    s = Strategy(op_prices, high_prices, low_prices, close_prices)

    statistics, trades, return_y = s.run()

    print(statistics)
    print(trades[:]['profit'])

    plot(op_prices, high_prices, low_prices, close_prices, trades, s.lower, s.sma, s.upper, return_y)


def run():
    data = pd.read_csv('../../data/EURUSD_5m_2024-01-26_99k.csv')
    data = data[-10000:]

    backtest(data)


if __name__ == '__main__':
    run()
