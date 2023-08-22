### also extract volume features

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from config import *
from utils import *
# Implementing the changes for volume features extraction based on the highs and lows from close prices
# Volumen feature extractor functions

def mean_value(df, column_name):
    """Calculate the mean value of a column."""
    return df[column_name].mean()

def median_value(df, column_name):
    """Calculate the median value of a column."""
    return df[column_name].median()

def variance_value(df, column_name):
    """Calculate the variance of a column."""
    return df[column_name].var()

def std_dev_value(df, column_name):
    """Calculate the standard deviation of a column."""
    return df[column_name].std()

def skewness_value(df, column_name):
    """Calculate the skewness of a column."""
    return skew(df[column_name])

def kurtosis_value(df, column_name):
    """Calculate the kurtosis of a column."""
    return kurtosis(df[column_name])

def mean_pct_change(df, column_name):
    """Calculate the mean percentage change of a column."""
    return df[column_name].pct_change().mean()

def total_volume(df, column_name="Volume"):
    """Calculate the total volume for the segment."""
    return df[column_name].sum()


def mean_volume(df, column_name="Volume"):
    """Calculate the mean volume for the segment."""
    return df[column_name].mean()


def volume_variance(df, column_name="Volume"):
    """Calculate the variance of the volume for the segment."""
    return df[column_name].var()


def volume_std_dev(df, column_name="Volume"):
    """Calculate the standard deviation of the volume for the segment."""
    return df[column_name].std()


def volume_skewness(df, column_name="Volume"):
    """Calculate the skewness of the volume for the segment."""
    return skew(df[column_name])


def volume_kurtosis(df, column_name="Volume"):
    """Calculate the kurtosis of the volume for the segment."""
    return kurtosis(df[column_name])


def process_row_volume_updated(df, idx, lookback, feature_extractors, volume_feature_extractors, column_name="Close",
                               volume_column_name="Volume"):
    """Process a single row to extract features based on lookback for both price and volume.

    Parameters:
    - df: DataFrame containing the data
    - idx: Index of the row to process
    - lookback: Length of the lookback period
    - feature_extractors: List of feature extraction functions for price
    - column_name: Name of the column to extract price features from
    - volume_column_name: Name of the column to extract volume features from

    Returns:
    - features: Extracted features for the row
    """
    # Get the lookback window for the current index
    start_idx = max(0, idx - lookback)
    window_df = df.iloc[start_idx:idx].copy()

    # Reset the indices of the window dataframe
    window_df.reset_index(drop=True, inplace=True)

    # Normalize the Close prices within the window
    min_val = window_df[column_name].min()
    max_val = window_df[column_name].max()
    window_df["Normalized"] = (window_df[column_name] - min_val) / (max_val - min_val)

    # Find refined highs and lows
    highs = refine_highs(window_df, "Normalized", 4)
    lows = refine_lows(window_df, "Normalized", 4)

    # Combine and sort highs and lows to create segments
    extremas = sorted(highs + lows)[-4:]  # Only consider the last 4 extremas

    # Extract features for each segment
    all_features = []
    for i in range(len(extremas) - 1):
        segment = window_df.iloc[extremas[i]:extremas[i + 1]]
        features = extract_segment_features(segment, feature_extractors)
        volume_features = extract_segment_features(segment, volume_feature_extractors, column_name=volume_column_name)
        all_features.extend(features + volume_features)

    # Include the segment from the last extrema to the end of the lookback window
    if extremas:
        segment = window_df.iloc[extremas[-1]:]
        features = extract_segment_features(segment, feature_extractors)
        volume_features = extract_segment_features(segment, volume_feature_extractors, column_name=volume_column_name)
        all_features.extend(features + volume_features)

    ## add relative prices to extrema
    rel_prices = [window_df.iloc[ex]["Normalized"] - window_df.iloc[-1]["Normalized"] for ex in extremas]
    all_features.extend(rel_prices)

    ## extrema values
    extrema_prices = [window_df.iloc[ex]["Normalized"] for ex in extremas]
    all_features.extend(extrema_prices)

    # If no segments were found, return NaN
    if not all_features:
        return [None] * len(extremas) * (
                    (len(feature_extractors) + len(volume_feature_extractors)) + len(rel_prices) + len(extrema_prices))

    return all_features


# Provided utility functions
def find_highs(df, column_name):
    highs = []
    if df[column_name].iloc[0] > df[column_name].iloc[1]:
        highs.append(df.index[0])
    for i in range(1, len(df) - 1):
        if df[column_name].iloc[i] > df[column_name].iloc[i - 1] and df[column_name].iloc[i] > df[column_name].iloc[
            i + 1]:
            highs.append(df.index[i])
    if df[column_name].iloc[-1] > df[column_name].iloc[-2]:
        highs.append(df.index[-1])
    return highs


def find_lows(df, column_name):
    lows = []
    if df[column_name].iloc[0] < df[column_name].iloc[1]:
        lows.append(df.index[0])
    for i in range(1, len(df) - 1):
        if df[column_name].iloc[i] < df[column_name].iloc[i - 1] and df[column_name].iloc[i] < df[column_name].iloc[
            i + 1]:
            lows.append(df.index[i])
    if df[column_name].iloc[-1] < df[column_name].iloc[-2]:
        lows.append(df.index[-1])
    return lows


def refine_highs(df, column_name, target_length, min_length=2):
    refined_highs = find_highs(df, column_name)
    while len(refined_highs) > target_length:
        last_refined_highs = refined_highs
        refined_highs = find_highs(df.iloc[refined_highs], column_name)
        if len(refined_highs) < min_length:
            refined_highs = last_refined_highs
            break
    return refined_highs


def refine_lows(df, column_name, target_length, min_length=2):
    refined_lows = find_lows(df, column_name)
    while len(refined_lows) > target_length:
        last_refined_lows = refined_lows
        refined_lows = find_lows(df.iloc[refined_lows], column_name)
        if len(refined_lows) < min_length:
            refined_lows = last_refined_lows
            break
    return refined_lows


# The provided code
def extract_segment_features(segment, feature_extractors, column_name="Close"):
    return [feature_extractor(segment, column_name) for feature_extractor in feature_extractors]


def extract_features_from_df(df, lookbacks=[25, 250, 1000], feature_extractors=[mean_pct_change],
                                     volume_feature_extractors=[], column_name="Close"):
    features_data = []
    for i in range(len(df)):
        row_features = {}
        for lookback in lookbacks:
            if i <= lookback:
                continue
            features = process_row_volume_updated(df, i, lookback, feature_extractors, volume_feature_extractors,
                                                  column_name)
            for j, feature in enumerate(features, 1):
                row_features[f"Feature_{lookback}_{j}"] = feature
        features_data.append(row_features)
    features_df = pd.DataFrame(features_data)
    augmented_df = pd.concat([df, features_df], axis=1)
    return augmented_df


# Extended list of feature extractor functions for volume
volume_feature_extractors = [
    total_volume,
    mean_volume
]
feature_extractors_extended = [
    mean_pct_change,
    std_dev_value,
]

def extract_last_feature(df):
    """ extracts features for last row in df
    returns df with features of last row added
    """

    # !TODO add length assertions
    df_ = df.iloc[-int(max(feature_lookbacks)*1.2):].reset_index(drop=True)
    df_feats = extract_features_from_df(df_, lookbacks=feature_lookbacks, feature_extractors=feature_extractors_extended,volume_feature_extractors=volume_feature_extractors, column_name="Close")
    df_feats = df_feats.iloc[-1:]
    merged_test =merge_df_with_feature_df(df, df_feats)
    return merged_test

## example
## augmented_df_volume_updated = extract_features_from_df(df, lookbacks=[6, 24, 4*24, 4*4*24, 4*4*24], feature_extractors=feature_extractors_extended, volume_feature_extractors=volume_feature_extractors)
