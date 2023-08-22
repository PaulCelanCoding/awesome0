## adds last TRAIN_WINDOW_SIZE to original df

from config import TRAIN_WINDOW_SIZE
from utils import get_all_features
from extract_ftrs import *
from config import feature_lookbacks, TRAIN_WINDOW_SIZE

# 3. Loop over the DataFrame
def integrate_features_to_df(df):
    """
    Integrate the last TRAIN_WINDOW_SIZE features to the original dataframe.

    Parameters:
    - df: The original dataframe.

    Returns:
    - final_concatted_df: The dataframe with integrated features.
    """

    # Extract features for the last TRAIN_WINDOW_SIZE rows
    feature_data = []
    for i in range(-TRAIN_WINDOW_SIZE, 0):
        df_ = df.iloc[i - int(max(feature_lookbacks) * 1.2): i].reset_index(drop=True)
        df_feats = extract_features_from_df(df_, feature_lookbacks, feature_extractors_extended, volume_feature_extractors, column_name="Close")
        all_feats = get_all_features(df_feats).iloc[-1].values
        feature_data.append(all_feats)

    # Convert the feature data to a DataFrame and replace NaN values with 0
    feature_df = pd.DataFrame(feature_data, columns=[f"Feature_{i}" for i in range(len(all_feats))])
    feature_df = feature_df.replace(np.NaN, 0)

    # Concatenate the original dataframe with the new feature DataFrame
    non_feature_cols = [col for col in df.columns if col not in feature_df.columns]
    nan_features = pd.DataFrame(np.nan, index=df.index, columns=feature_df.columns)
    initial_data = pd.concat([df[non_feature_cols], nan_features], axis=1)
    initial_data.iloc[-TRAIN_WINDOW_SIZE:, -len(feature_df.columns):] = feature_df.values

    return initial_data