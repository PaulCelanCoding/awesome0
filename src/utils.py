import pandas as pd


def get_all_features(df):
    """
    Extracts all the columns containing features from the DataFrame.

    Parameters:
    - df: The DataFrame containing the data.

    Returns:
    - A DataFrame containing only the feature columns.
    """
    # Extract columns containing "Feature" in the column name
    features = df.columns[df.columns.str.contains("Feature")]
    return df[features]

def rename_for_freqtrade(df):
    # Define the renaming dictionary
    renaming_dict = {
        'date': 'Open_time',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }

    # Create renamed columns and append them to the original DataFrame
    for original, new in renaming_dict.items():
        df[new] = df[original]

    return df


def merge_df_with_feature_df(base_df, feats_df):
    """
    Merges two dataframes on the "date" column, using base_df as the base.
    Duplicates, other than the "date" column, are removed and missing values are filled with NaN.

    Parameters:
    - base_df: The main dataframe to be used as the base.
    - additional_df: The additional dataframe whose columns will be added to the base dataframe.

    Returns:
    - A merged dataframe.
    """
    # Merge the dataframes using base_df as the base
    merged_df = pd.merge(base_df, feats_df, on="date", suffixes=('', '_additional'), how='left')

    # Drop the columns with suffix "_additional" that are duplicates
    to_drop = [col for col in merged_df if col.endswith('_additional')]
    merged_df = merged_df.drop(columns=to_drop)

    return merged_df
