import numpy as np
import pandas as pd


atr_plus = 2
atr_minus = 1

# Function to calculate the True Range
def true_range(high, low, prev_close):
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def label_data(row, df, atr_plus, atr_minus):
    current_close = row['Close']
    current_atr = row['atr']
    future_closes = df.loc[row.name + 1:, 'Close'].values

    # Check for the condition: Close price rises by 2 ATR before falling by 1 ATR
    has_risen_2_atr = False
    for future_close in future_closes:
        price_diff = future_close - current_close
        if price_diff <= -atr_minus * current_atr:
            return 0
        if price_diff >= atr_plus * current_atr:
            return 1
    return np.nan  # If no condition is met, return NaN


def add_labels(df):
    # Calculate True Range and ATR
    df['true_range'] = df.apply(
        lambda row: true_range(row['High'], row['Low'], df.loc[row.name - 1, 'Close'] if row.name > 0 else 0), axis=1)
    atr_period = 14
    df['atr'] = df['true_range'].rolling(window=atr_period).mean()

    # Label the data
    df['label'] = df.apply(lambda row: label_data(row, df, atr_plus, atr_minus), axis=1)

    return df


