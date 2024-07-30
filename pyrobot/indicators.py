import numpy as np
import pandas as pd

from typing import Any
from typing import Dict
from typing import Union

from pyrobot.stock_frame import StockFrame

class Indicators():

       
    
    def __init__(self, price_data_frame: StockFrame) -> None:
        

        self._stock_frame: StockFrame = price_data_frame
        self._price_groups = price_data_frame.symbol_groups
        self._current_indicators = {}
        self._indicator_signals = {}
        self._frame = self._stock_frame.frame

        self._indicators_comp_key = []
        self._indicators_key = []
        
        if self.is_multi_index:
            True

    def get_indicator_signal(self, indicator: str= None) -> Dict:
        

        if indicator and indicator in self._indicator_signals:
            return self._indicator_signals[indicator]
        else:      
            return self._indicator_signals
    
    def set_indicator_signal(self, indicator: str, buy: float, sell: float, condition_buy: Any, condition_sell: Any, 
                             buy_max: float = None, sell_max: float = None, condition_buy_max: Any = None, condition_sell_max: Any = None) -> None:
        

        # Add the key if it doesn't exist.
        if indicator not in self._indicator_signals:
            self._indicator_signals[indicator] = {}
            self._indicators_key.append(indicator)      

        # Add the signals.
        self._indicator_signals[indicator]['buy'] = buy     
        self._indicator_signals[indicator]['sell'] = sell
        self._indicator_signals[indicator]['buy_operator'] = condition_buy
        self._indicator_signals[indicator]['sell_operator'] = condition_sell

        # Add the max signals
        self._indicator_signals[indicator]['buy_max'] = buy_max  
        self._indicator_signals[indicator]['sell_max'] = sell_max
        self._indicator_signals[indicator]['buy_operator_max'] = condition_buy_max
        self._indicator_signals[indicator]['sell_operator_max'] = condition_sell_max

    def set_indicator_signal_compare(self, indicator_1: str, indicator_2: str, condition_buy: Any, condition_sell: Any) -> None:
        

        # Define the key.
        key = "{ind_1}_comp_{ind_2}".format(
            ind_1=indicator_1,
            ind_2=indicator_2
        )

        # Add the key if it doesn't exist.
        if key not in self._indicator_signals:
            self._indicator_signals[key] = {}
            self._indicators_comp_key.append(key)   

        # Grab the dictionary.
        indicator_dict = self._indicator_signals[key]

        # Add the signals.
        indicator_dict['type'] = 'comparison'
        indicator_dict['indicator_1'] = indicator_1
        indicator_dict['indicator_2'] = indicator_2
        indicator_dict['buy_operator'] = condition_buy
        indicator_dict['sell_operator'] = condition_sell

    @property
    def price_data_frame(self) -> pd.DataFrame:
       
        return self._frame

    @price_data_frame.setter
    def price_data_frame(self, price_data_frame: pd.DataFrame) -> None:
        

        self._frame = price_data_frame

    @property
    def is_multi_index(self) -> bool:
        

        if isinstance(self._frame.index, pd.MultiIndex):
            return True
        else:
            return False

    def change_in_price(self, column_name: str = 'change_in_price') -> pd.DataFrame:
        

        locals_data = locals()
        del locals_data['self']
        
        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.change_in_price

        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.diff()
        )

        return self._frame

    def rsi(self, period: int, method: str = 'wilders', column_name: str = 'rsi') -> pd.DataFrame:
        

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.rsi

        # First calculate the Change in Price.
        if 'change_in_price' not in self._frame.columns:
            self.change_in_price()

        # Define the up days.
        self._frame['up_day'] = self._price_groups['change_in_price'].transform(
            lambda x : np.where(x >= 0, x, 0)
        )

        # Define the down days.
        self._frame['down_day'] = self._price_groups['change_in_price'].transform(
            lambda x : np.where(x < 0, x.abs(), 0)
        )

        # Calculate the EWMA for the Up days.
        self._frame['ewma_up'] = self._price_groups['up_day'].transform(
            lambda x: x.ewm(span = period).mean()
        )

        # Calculate the EWMA for the Down days.
        self._frame['ewma_down'] = self._price_groups['down_day'].transform(
            lambda x: x.ewm(span = period).mean()
        )

        # Calculate the Relative Strength
        relative_strength = self._frame['ewma_up'] / self._frame['ewma_down']

        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

        # Add the info to the data frame.
        self._frame['rsi'] = np.where(relative_strength_index == 0, 100, 100 - (100 / (1 + relative_strength_index)))

        # Clean up before sending back.
        self._frame.drop(
            labels=['ewma_up', 'ewma_down', 'down_day', 'up_day', 'change_in_price'],
            axis=1,
            inplace=True
        )

        return self._frame

    def sma(self, period: int, column_name: str = 'sma') -> pd.DataFrame:
        

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.sma

        # Add the SMA
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        return self._frame

    def ema(self, period: int, alpha: float = 0.0, column_name = 'ema') -> pd.DataFrame:
      

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.ema

        # Add the EMA
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.ewm(span=period).mean()
        )

        return self._frame

    def rate_of_change(self, period: int = 1, column_name: str = 'rate_of_change') -> pd.DataFrame:
        
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.rate_of_change

        # Add the Momentum indicator.
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.pct_change(periods=period)
        )

        return self._frame        

    def bollinger_bands(self, period: int = 20, column_name: str = 'bollinger_bands') -> pd.DataFrame:
       
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.bollinger_bands

        # Define the Moving Avg.
        self._frame['moving_avg'] = self._price_groups['close'].transform(
            lambda x : x.rolling(window=period).mean()
        )

        # Define Moving Std.
        self._frame['moving_std'] = self._price_groups['close'].transform(
            lambda x : x.rolling(window=period).std()
        )

        # Define the Upper Band.
        self._frame['band_upper'] = 4 * (self._frame['moving_std'] / self._frame['moving_avg'])

        # Define the lower band
        self._frame['band_lower'] = (
            (self._frame['close'] - self._frame['moving_avg']) + 
            (2 * self._frame['moving_std']) / 
            (4 * self._frame['moving_std'])
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['moving_avg', 'moving_std'],
            axis=1,
            inplace=True
        )

        return self._frame   

    def average_true_range(self, period: int = 14, column_name: str ='average_true_range') -> pd.DataFrame:
      
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.average_true_range


        # Calculate the different parts of True Range.
        self._frame['true_range_0'] = abs(self._frame['high'] - self._frame['low'])
        self._frame['true_range_1'] = abs(self._frame['high'] - self._frame['close'].shift())
        self._frame['true_range_2'] = abs(self._frame['low'] - self._frame['close'].shift())

        # Grab the Max.
        self._frame['true_range'] = self._frame[['true_range_0', 'true_range_1', 'true_range_2']].max(axis=1)

        # Calculate the Average True Range.
        self._frame['average_true_range'] = self._frame['true_range'].transform(
            lambda x: x.ewm(span = period, min_periods = period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['true_range_0', 'true_range_1', 'true_range_2', 'true_range'],
            axis=1,
            inplace=True
        )

        return self._frame

    def stochastic_oscillator(self, column_name: str = 'stochastic_oscillator') -> pd.DataFrame:
       

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.stochastic_oscillator

        # Calculate the stochastic_oscillator.
        self._frame['stochastic_oscillator'] = (
            self._frame['close'] - self._frame['low'] / 
            self._frame['high'] - self._frame['low']
        )

        return self._frame 

    def macd(self, fast_period: int = 12, slow_period: int = 26, column_name: str = 'macd') -> pd.DataFrame:
        

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.macd

        # Calculate the Fast Moving MACD.
        self._frame['macd_fast'] = self._frame['close'].transform(
            lambda x: x.ewm(span = fast_period, min_periods = fast_period).mean()
        )

        # Calculate the Slow Moving MACD.
        self._frame['macd_slow'] = self._frame['close'].transform(
            lambda x: x.ewm(span = slow_period, min_periods = slow_period).mean()
        )

        # Calculate the difference between the fast and the slow.
        self._frame['macd_diff'] = self._frame['macd_fast'] - self._frame['macd_slow']

        # Calculate the Exponential moving average of the fast.
        self._frame['macd'] = self._frame['macd_diff'].transform(
            lambda x: x.ewm(span = 9, min_periods = 8).mean()
        )

        return self._frame 

    def mass_index(self, period: int = 9, column_name: str = 'mass_index') -> pd.DataFrame:
        

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.mass_index

        # Calculate the Diff.
        self._frame['diff'] = self._frame['high'] - self._frame['low']

        # Calculate Mass Index 1
        self._frame['mass_index_1'] = self._frame['diff'].transform(
            lambda x: x.ewm(span = period, min_periods = period - 1).mean()
        )

        # Calculate Mass Index 2
        self._frame['mass_index_2'] = self._frame['mass_index_1'].transform(
            lambda x: x.ewm(span = period, min_periods = period - 1).mean()
        )
        
        # Grab the raw index.
        self._frame['mass_index_raw'] = self._frame['mass_index_1'] / self._frame['mass_index_2']

        # Calculate the Mass Index.
        self._frame['mass_index'] = self._frame['mass_index_raw'].transform(
            lambda x: x.rolling(window=25).sum()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['diff', 'mass_index_1', 'mass_index_2', 'mass_index_raw'],
            axis=1,
            inplace=True
        )

        return self._frame
    
    def force_index(self, period: int, column_name: str = 'force_index') -> pd.DataFrame:
        

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.force_index

        # Calculate the Force Index.
        self._frame[column_name] = self._frame['close'].diff(period)  * self._frame['volume'].diff(period)

        return self._frame

    def ease_of_movement(self, period: int, column_name: str = 'ease_of_movement') -> pd.DataFrame:
        


        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.ease_of_movement
        
        # Calculate the ease of movement.
        high_plus_low = (self._frame['high'].diff(1) + self._frame['low'].diff(1))
        diff_divi_vol = (self._frame['high'] - self._frame['low']) / (2 * self._frame['volume'])
        self._frame['ease_of_movement_raw'] = high_plus_low * diff_divi_vol

        # Calculate the Rolling Average of the Ease of Movement.
        self._frame['ease_of_movement'] = self._frame['ease_of_movement_raw'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['ease_of_movement_raw'],
            axis=1,
            inplace=True
        )

        return self._frame

    def commodity_channel_index(self, period: int, column_name: str = 'commodity_channel_index') -> pd.DataFrame:
       

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.commodity_channel_index

        # Calculate the Typical Price.
        self._frame['typical_price'] = (self._frame['high'] + self._frame['low'] + self._frame['close']) / 3

        # Calculate the Rolling Average of the Typical Price.
        self._frame['typical_price_mean'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Calculate the Rolling Standard Deviation of the Typical Price.
        self._frame['typical_price_std'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).std()
        )

        # Calculate the Commodity Channel Index.
        self._frame[column_name] = self._frame['typical_price_mean'] / self._frame['typical_price_std']

        # Clean up before sending back.
        self._frame.drop(
            labels=['typical_price', 'typical_price_mean', 'typical_price_std'],
            axis=1,
            inplace=True
        )

        return self._frame

    def standard_deviation(self, period: int, column_name: str = 'standard_deviation') -> pd.DataFrame:
        

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.standard_deviation

        # Calculate the Standard Deviation.
        self._frame[column_name] = self._frame['close'].transform(
            lambda x: x.ewm(span=period).std()
        )

        return self._frame

    def chaikin_oscillator(self, period: int, column_name: str = 'chaikin_oscillator') -> pd.DataFrame:
      

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.chaikin_oscillator

        # Calculate the Money Flow Multiplier.
        money_flow_multiplier_top = 2 * (self._frame['close'] - self._frame['high'] - self._frame['low'])
        money_flow_multiplier_bot = (self._frame['high'] - self._frame['low'])

        # Calculate Money Flow Volume
        self._frame['money_flow_volume'] = (money_flow_multiplier_top / money_flow_multiplier_bot) * self._frame['volume']

        # Calculate the 3-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_3'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=3, min_periods=2).mean()
        )

        # Calculate the 10-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_10'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=10, min_periods=9).mean()
        )

        # Calculate the Chaikin Oscillator.
        self._frame[column_name] = self._frame['money_flow_volume_3'] - self._frame['money_flow_volume_10']

        # Clean up before sending back.
        self._frame.drop(
            labels=['money_flow_volume_3', 'money_flow_volume_10', 'money_flow_volume'],
            axis=1,
            inplace=True
        )

        return self._frame

    def kst_oscillator(self, r1: int, r2: int, r3: int, r4: int, n1: int, n2: int, n3: int, n4: int, column_name: str = 'kst_oscillator') -> pd.DataFrame:
       

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.kst_oscillator

        # Calculate the ROC 1.
        self._frame['roc_1'] = self._frame['close'].diff(r1 - 1)  / self._frame['close'].shift(r1 - 1)

        # Calculate the ROC 2.
        self._frame['roc_2'] = self._frame['close'].diff(r2 - 1)  / self._frame['close'].shift(r2 - 1)

        # Calculate the ROC 3.
        self._frame['roc_3'] = self._frame['close'].diff(r3 - 1)  / self._frame['close'].shift(r3 - 1)

        # Calculate the ROC 4.
        self._frame['roc_4'] = self._frame['close'].diff(r4 - 1)  / self._frame['close'].shift(r4 - 1)


        # Calculate the Mass Index.
        self._frame['roc_1_n'] = self._frame['roc_1'].transform(
            lambda x: x.rolling(window=n1).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_2_n'] = self._frame['roc_2'].transform(
            lambda x: x.rolling(window=n2).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_3_n'] = self._frame['roc_3'].transform(
            lambda x: x.rolling(window=n3).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_4_n'] = self._frame['roc_4'].transform(
            lambda x: x.rolling(window=n4).sum()
        )

        self._frame[column_name] = 100 * (self._frame['roc_1_n'] + 2 * self._frame['roc_2_n'] + 3 * self._frame['roc_3_n'] + 4 * self._frame['roc_4_n'])
        self._frame[column_name + "_signal"] = self._frame['column_name'].transform(
            lambda x: x.rolling().mean()
        )
        
        # Clean up before sending back.
        self._frame.drop(
            labels=['roc_1', 'roc_2', 'roc_3', 'roc_4', 'roc_1_n', 'roc_2_n', 'roc_3_n', 'roc_4_n'],
            axis=1,
            inplace=True
        )

        return self._frame

    def refresh(self):
        """Updates the Indicator columns after adding the new rows."""

        # First update the groups since, we have new rows.
        self._price_groups = self._stock_frame.symbol_groups

        # Grab all the details of the indicators so far.
        for indicator in self._current_indicators:
            
            # Grab the function.
            indicator_argument = self._current_indicators[indicator]['args']

            # Grab the arguments.
            indicator_function = self._current_indicators[indicator]['func']

            # Update the function.
            indicator_function(**indicator_argument)

    def check_signals(self) -> Union[pd.DataFrame, None]:
       

        signals_df = self._stock_frame._check_signals(
            indicators=self._indicator_signals,
            indciators_comp_key=self._indicators_comp_key,
            indicators_key=self._indicators_key
        )

        return signals_df


