import pandas as pd
import logging
from typing import Dict
from config import BacktestConfig
from strategies.base import Strategy

class BacktestEngine:
    """Simulates trading strategy execution against historical data."""
    def __init__(self,config:BacktestConfig,strategy:Strategy):
        self.config = config
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

        self.cash = self.config.initial_capital
        self.position = {}
        self.trade_log = []
        self.equity_curve = []
        
    def _check_stops(self,symbol:str,date:str,all_signals:dict,active_orders:dict,highest_closes:dict,lowest_closes:dict) -> None:
        """Check stop-loss, trailing stop, and take-profit for an open position."""
        if symbol not in all_signals or date not in all_signals[symbol].index:
            return
        row = all_signals[symbol].loc[date]
        price = row['close']
        low = row.get('low', price)
        high = row.get('high', price)

        if symbol not in active_orders or active_orders[symbol] is None:
            return

        order = active_orders[symbol]
        direction = self.position[symbol]['direction']

        if direction == 'long':
            highest_closes[symbol] = max(highest_closes.get(symbol, 0), price)

            stops = []
            if order.stop_loss is not None:
                stops.append(order.stop_loss)
            if order.trail_offset is not None:
                trail_stop = highest_closes[symbol] - order.trail_offset
                stops.append(trail_stop)
                if order.stop_loss is None or trail_stop > order.stop_loss:
                    order.stop_loss = trail_stop

            # Long stop: price drops below stop level
            if stops and low <= max(stops):
                self._execute_sell(symbol, date, max(stops))
                active_orders.pop(symbol, None)
                highest_closes.pop(symbol, None)
                return

            # Long take-profit: price rises above target
            if order.take_profit is not None and price >= order.take_profit:
                self._execute_sell(symbol, date, order.take_profit)
                active_orders.pop(symbol, None)
                highest_closes.pop(symbol, None)
                return

        elif direction == 'short':
            # Track lowest close since entry for trailing stop
            lowest_closes[symbol] = min(lowest_closes.get(symbol,float('inf')),price)

            stops = []
            if order.stop_loss is not None:
                stops.append(order.stop_loss)
            if order.trail_offset is not None:
                # Trailing stop sits above lowest close
                trail_stop = lowest_closes[symbol] + order.trail_offset
                stops.append(trail_stop)
                # Ratchet stop down as price drops (locks in profit)
                if order.stop_loss is None or trail_stop < order.stop_loss:
                    order.stop_loss = trail_stop

            # Short stop: price rises above stop level
            if stops and high >= min(stops):
                self._execute_cover(symbol,date,min(stops))
                active_orders.pop(symbol,None)
                lowest_closes.pop(symbol,None)
                return

            # Short take-profit: price drops below target
            if order.take_profit is not None and price <= order.take_profit:
                self._execute_cover(symbol,date,order.take_profit)
                active_orders.pop(symbol,None)
                lowest_closes.pop(symbol,None)
                return

    def _execute_signals(self,symbol:str,date:str,sig_df:pd.DataFrame,active_orders:dict,all_orders:dict,highest_closes:dict,lowest_closes:dict) -> None:
        """Route strategy signals to the appropriate execution method."""
        if date not in sig_df.index:
            return
        row = sig_df.loc[date]
        price = row['close']
        signal = row['signal']
        orig_idx = row['_orig_idx']

        if signal == 'buy' and symbol not in self.position:
            if len(self.position) < self.config.max_positions:
                self._execute_buy(symbol, date, price)
                if symbol in self.position:
                    active_orders[symbol] = all_orders[symbol].get(orig_idx)
                    highest_closes[symbol] = price

        elif signal == 'sell' and symbol in self.position and self.position[symbol]['direction'] == 'long':
            self._execute_sell(symbol, date, price)
            active_orders.pop(symbol, None)
            highest_closes.pop(symbol, None)

        elif signal == 'short' and symbol not in self.position:
            if len(self.position) < self.config.max_positions:
                self._execute_short(symbol,date,price)
                if symbol in self.position:
                    active_orders[symbol] = all_orders[symbol].get(orig_idx)
                    lowest_closes[symbol] = price

        elif signal == 'cover' and symbol in self.position and self.position[symbol]['direction'] == 'short':
            self._execute_cover(symbol,date,price)
            active_orders.pop(symbol,None)
            lowest_closes.pop(symbol,None)

    def _record_equity(self,symbol:str,all_signals:dict,date:str,pos:dict) -> float:
        """Return the equity contribution of a single position."""
        if symbol in all_signals and date in all_signals[symbol].index:
            current_price = all_signals[symbol].loc[date]['close']
            if pos['direction'] == 'long':
                return pos['shares'] * current_price
            else:
                # Short equity: original proceeds + unrealized P&L
                return pos['shares'] * (2 * pos['entry_price'] - current_price)
        else:
            return pos['shares'] * pos['entry_price']

    def _force_close(self,symbol:str,all_signals:dict) -> None:
        """Force close a position at end of data."""
        sig_df = all_signals[symbol]
        last = sig_df.iloc[-1]
        last_date = sig_df.index[-1]
        direction = self.position[symbol]['direction']
        self.logger.info(f"Force closing {symbol} {direction} position at end of data on {last_date}")
        if direction == 'long':
            self._execute_sell(symbol, last_date, last['close'])
        else:
            self._execute_cover(symbol, last_date, last['close'])

    def run(self,data:Dict[str,pd.DataFrame]) -> dict:
        """Execute strategy signals against historical data and return results."""

        # Backward compatibility: if values are DataFrames, wrap as single symbol
        first_val = next(iter(data.values()))
        if isinstance(first_val, pd.DataFrame):
            data = {'_single': data}

        # Pre-generate signals for all symbols
        all_signals = {}
        all_orders = {}

        for symbol, symbol_data in data.items():
            signals,orders = self.strategy.generate_signals(symbol_data)
            signals = signals.copy()
            signals['_orig_idx'] = signals.index
            all_signals[symbol] = signals.set_index('date')
            all_orders[symbol] = orders

        # Build unified date index
        all_dates = sorted(set().union(*(df.index for df in all_signals.values())))

        active_orders = {}
        highest_closes = {}
        lowest_closes = {}

        for date in all_dates:
            # Phase 1: Check stops/exits for open positions
            for symbol in list(self.position.keys()):
                self._check_stops(symbol, date, all_signals, active_orders, highest_closes, lowest_closes)

            # Phase 2: Execute new signals
            for symbol, sig_df in all_signals.items():
                self._execute_signals(symbol,date,sig_df,active_orders,all_orders,highest_closes,lowest_closes)

            # Record equity
            equity = self.cash
            for symbol, pos in self.position.items():
                equity += self._record_equity(symbol,all_signals,date,pos)
            self.equity_curve.append({'date': date, 'equity': equity})

        # Force close any remaining positions
        for symbol in list(self.position.keys()):
            self._force_close(symbol,all_signals)

        # Computing basic results from backtest
        results = self._build_results()

        # Reset initial values
        self.cash = self.config.initial_capital
        self.position = {}
        self.trade_log = []
        self.equity_curve = []

        return results

    def _build_results(self) -> dict:
        """Package trade log"""

        equity_df = pd.DataFrame(self.equity_curve)
        wins = [trade for trade in self.trade_log if trade['pnl'] > 0]
        losses = [trade for trade in self.trade_log if trade['pnl'] < 0]
        total_pnl = sum(trade['pnl'] for trade in self.trade_log)

        # Max Drawdown
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()

        return {
            'trade_log':self.trade_log,
            'equity_curve': equity_df,
            'summary': {
                'total_trades': len(self.trade_log),
                'winning_trades':len(wins),
                'losing_trades':len(losses),
                'win_rate':len(wins) / len(self.trade_log) if self.trade_log else 0,
                'total_pnl': total_pnl,
                'total_return_pct': total_pnl / self.config.initial_capital,
                'max_drawdown': max_drawdown,
                'final_equity': equity_df['equity'].iloc[-1],
            }
        }

    def _execute_buy(self,symbol:str,date:str,price:float):
        """Open a long position for a symbol with equal allocation sizing."""

        budget = self.config.initial_capital / self.config.max_positions
        fill_price = self._apply_slippage(price,'buy')
        shares = int(budget // fill_price)

        if shares <= 0:
            self.logger.warning(f"Insufficient budget to buy {symbol} at {price:.2f} on {date}")
            return

        cost = shares * fill_price
        commission = self._calculate_commission(cost)

        if cost + commission > self.cash:
            self.logger.warning(f"Insufficient cash for {symbol}: need {cost+commission:.2f}, have {self.cash:.2f}")
            return

        self.cash -= (cost + commission)
        self.position[symbol] = {
            'shares': shares,
            'entry_price':fill_price,
            'entry_date':date,
            'commission_in':commission,
            'direction': 'long'
            }

    def _execute_sell(self,symbol:str,date:str,price:float):
        """Close the open position at the given price with costs applied."""
        pos = self.position[symbol]
        fill_price = self._apply_slippage(price,'sell')
        proceeds = pos['shares'] * fill_price
        commission = self._calculate_commission(proceeds)

        self.cash += (proceeds - commission)

        # Calculate P&L
        entry_cost = pos['shares'] * pos['entry_price']
        total_commission = pos['commission_in'] + commission
        pnl = proceeds - entry_cost - total_commission

        # Append to trade log
        self.trade_log.append({
            'symbol': symbol,
            'entry_date':pos['entry_date'],
            'exit_date':date,
            'entry_price':pos['entry_price'],
            'exit_price': fill_price,
            'shares': pos['shares'],
            'direction': 'long',
            'pnl':pnl,
            'return_pct':pnl / entry_cost,
            'commission_paid': total_commission,
        })

        del self.position[symbol]

    def _execute_short(self,symbol:str,date:str,price:float):
        """Open a short position for a symbol (borrow and sell shares)."""

        budget = self.config.initial_capital / self.config.max_positions
        fill_price = self._apply_slippage(price,'sell')
        shares = int(budget // fill_price)

        if (shares <= 0):
            self.logger.warning(f"Insufficient budget to short {symbol} at {price:.2f} on {date}")
            return

        proceeds = shares * fill_price
        commission = self._calculate_commission(proceeds)

        if commission > self.cash:
            self.logger.warning(f"Insufficient cash for {symbol} short commission: need {commission:.2f}, have {self.cash:.2f}")
            return

        # Cash increases from selling borrowed shares
        self.cash += (proceeds - commission)
        self.position[symbol] = {
            'shares': shares,
            'entry_price':fill_price,
            'entry_date':date,
            'commission_in':commission,
            'direction': 'short'
            }

    def _execute_cover(self,symbol:str,date:str,price:float):
        """Close a short position by buying back shares."""
        pos = self.position[symbol]
        fill_price = self._apply_slippage(price,'buy')
        cost = pos['shares'] * fill_price
        commission = self._calculate_commission(cost)

        # Cash decreases from buying back shares
        self.cash -= (cost + commission)

        # P&L: profit when cover price < entry price
        entry_proceeds = pos['shares'] * pos['entry_price']
        total_commission = pos['commission_in'] + commission
        pnl = entry_proceeds - cost - total_commission

        # Append to trade log
        self.trade_log.append({
            'symbol': symbol,
            'entry_date':pos['entry_date'],
            'exit_date':date,
            'entry_price':pos['entry_price'],
            'exit_price': fill_price,
            'shares': pos['shares'],
            'direction': 'short',
            'pnl':pnl,
            'return_pct':pnl / entry_proceeds,
            'commission_paid': total_commission,
        })

        del self.position[symbol]

    def _apply_slippage(self,stock_price:float,direction:str) -> float:
        if direction == 'buy':
            # Calculate share price after slippage
            price_after_slippage = stock_price + (stock_price * self.config.slippage)

            # Returns share price after slippage and spreads
            return self._apply_spreads(price_after_slippage,direction)
        else:
            # Calculate share price after slippage
            price_after_slippage = stock_price - (stock_price * self.config.slippage)

            # Returns share price after slippage and spreads
            return self._apply_spreads(price_after_slippage,direction)

    def _apply_spreads(self,price:float,direction:str) -> float:
        if direction == 'buy':
            return price + self.config.spreads
        else:
            return price - self.config.spreads
        

    def _calculate_commission(self,trade_value: float) -> float:
        return trade_value * self.config.commission_rate

