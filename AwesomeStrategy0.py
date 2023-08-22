print("DEBUG: Starting script")

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

print("DEBUG: Libraries imported")

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

import sys

src_repo = r"C:\Users\xyz\Desktop\bodi2\awesome0\src"
sys.path.append(src_repo)

from extract_ftrs import *
from utils import rename_for_freqtrade, merge_df_with_feature_df, get_all_features
from config import feature_lookbacks, CLF_DIR
from retrainSignalAndNovelty import load_recent_models

print("DEBUG: Custom imports done")

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib

print("DEBUG: Additional libraries imported")


class AwesomeStrategy0(IStrategy):
    print("DEBUG: Class definition started")

    INTERFACE_VERSION = 3
    timeframe = '1h'
    can_short: bool = False
    minimal_roi = {"0": 2}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 30
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        print("DEBUG: populate_indicators started")
        df = dataframe
        df["volume"] = df["volume"] * df["close"]
        if "Close" not in df.columns:
            df = rename_for_freqtrade(df)
        df = extract_last_feature(df)
        featLast = get_all_features(df.iloc[-1:])
        featLast.columns = [f"Feature_{i}" for i in range(len(featLast.columns))]
        featLast = featLast.fillna(0)
        df["signal"] = np.NaN
        df["novelty"] = np.NaN
        df["entrycond"] = False
        df.loc[len(df) - 1, "signal"] = self.signal.predict(featLast) == 1
        df.loc[len(df) - 1, "novelty"] = self.novelty.predict(featLast) != 1
        df["entrycond"] = (df.novelty) & (df.signal)
        print("DEBUG: populate_indicators ended")
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        print("DEBUG: populate_entry_trend started")
        dataframe.loc[
            (
                dataframe['entrycond'].values
            ),
            'enter_long'] = 1
        print("DEBUG: populate_entry_trend ended")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        print("DEBUG: populate_exit_trend started")
        dataframe.loc[
            (
                ~dataframe['entrycond'].values
            ),
            'exit_long'] = 1
        print("DEBUG: populate_exit_trend ended")
        return dataframe

    def bot_start(self, **kwargs) -> None:
        print("DEBUG: bot_start started")
        self.signal, self.novelty = load_recent_models(CLF_DIR)
        print("DEBUG: bot_start ended")

print("DEBUG: Class definition ended")
