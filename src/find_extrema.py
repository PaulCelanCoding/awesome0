import os
import pandas as pd

def find_highs(df, column_name):
    """
    Corrected function to find high points based on specified criteria.

    Parameters:
    - df: DataFrame containing the data
    - column_name: Name of the column to search for highs

    Returns:
    - highs: List of indices where high points are located
    """

    highs = []

    # First point
    if df[column_name].iloc[0] > df[column_name].iloc[1]:
        highs.append(df.index[0])

    # Middle points
    for i in range(1, len(df) - 1):
        if df[column_name].iloc[i] > df[column_name].iloc[i - 1] and df[column_name].iloc[i] > df[column_name].iloc[
            i + 1]:
            highs.append(df.index[i])

    # Last point
    if df[column_name].iloc[-1] > df[column_name].iloc[-2]:
        highs.append(df.index[-1])

    return highs


def find_lows(df, column_name):
    """
    Corrected function to find low points based on specified criteria.

    Parameters:
    - df: DataFrame containing the data
    - column_name: Name of the column to search for lows

    Returns:
    - lows: List of indices where low points are located
    """

    lows = []

    # First point
    if df[column_name].iloc[0] < df[column_name].iloc[1]:
        lows.append(df.index[0])

    # Middle points
    for i in range(1, len(df) - 1):
        if df[column_name].iloc[i] < df[column_name].iloc[i - 1] and df[column_name].iloc[i] < df[column_name].iloc[
            i + 1]:
            lows.append(df.index[i])

    # Last point
    if df[column_name].iloc[-1] < df[column_name].iloc[-2]:
        lows.append(df.index[-1])

    return lows


def refine_highs(df, column_name, target_length, min_length=2):
    """
    Refines the high points until the desired target length is achieved.

    Parameters:
    - df: DataFrame containing the data
    - column_name: Name of the column to search for highs
    - target_length: Desired number of high points

    Returns:
    - refined_highs: List of indices where refined high points are located
    """

    refined_highs = find_highs(df, column_name)

    # Keep refining the high points until the desired length is achieved
    while len(refined_highs) > target_length:
        last_refined_highs = refined_highs
        refined_highs = find_highs(df.iloc[refined_highs], column_name)
        if len(refined_highs) < min_length:
            refined_highs = last_refined_highs
            break

    return refined_highs


def refine_lows(df, column_name, target_length, min_length=2):
    """
    Refines the low points until the desired target length is achieved.

    Parameters:
    - df: DataFrame containing the data
    - column_name: Name of the column to search for lows
    - target_length: Desired number of low points

    Returns:
    - refined_lows: List of indices where refined low points are located
    """

    refined_lows = find_lows(df, column_name)
    # Keep refining the low points until the desired length is achieved
    while len(refined_lows) > target_length:
        last_refined_lows = refined_lows
        refined_lows = find_lows(df.iloc[refined_lows], column_name)
        if len(refined_lows) < min_length:
            refined_lows = last_refined_lows
            break

    return refined_lows


