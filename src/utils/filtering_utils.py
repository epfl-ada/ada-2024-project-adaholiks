import numpy as np
import pandas as pd

def downsample_paths(df, threshold=10, seed=42):
    """
    Downsample to one all paths where the start target pair has been played by the same player multtiple times.
    Downsample the paths so that the a certain start/target artcle down not appear more than threshold times.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing path data.
    - threshold (int): The maximum number of times the same start-target pair can be played. Default is 10 (median).

    Returns:
    - downsampled_df (pd.DataFrame): The downsampled DataFrame.
    - num_removed (int): The number of paths removed.
    """

    # first downsample to 1 all paths where the start target pair has been played by the same player multtiple times 
    # (same IpAddress and identifier)
    df = df.groupby(['hashedIpAddress', 'identifier']).sample(n=1, random_state=42)

    # ------------------------------------

    # Downsample the paths so that the same start-target pair is not played more than the set threshold times

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Shuffle the DataFrame
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # first downfilter start articles that have more than threshold paths
    start_sampled = (
        shuffled_df.groupby('start_article')
        .head(threshold)
        .reset_index(drop=True)
    )
    # second downfilter target articles that have more than threshold paths
    downsampled_df = (
        start_sampled.groupby('target_article')
        .head(threshold)
        .reset_index(drop=True)
    )

    # the removed paths
    num_removed = df.shape[0] - downsampled_df.shape[0]

    return downsampled_df, num_removed


def IQR_filtering(df, column, lower_bound=False,  multiplier=1.5):
    """
    Filter the DataFrame based on the IQR method for each distance group. 
    The lower bound is the distance itself, and the upper bound is determined using the IQR method.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing path data.
    - column (str): The column to filter on.
    - multiplier (float): The multiplier for the IQR to determine the upper bound. Default is 1.5.

    Returns:
    - filtered_df (pd.DataFrame): The filtered DataFrame without outliers.
    - removed_count (int): The number of rows removed.
    - removed_percentage (float): The percentage of rows removed.
    """
    
    filtered_dfs = []  # List to hold filtered data for each distance group

    # Iterate over each unique distance
    for distance_value in df['distance'].unique():
        # Subset the DataFrame for the current distance group
        df_subset = df[df['distance'] == distance_value]

        # Compute IQR for the specified column
        Q1 = df_subset[column].quantile(0.25)
        Q3 = df_subset[column].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate the upper bound based on IQR
        upper_bound = Q3 + multiplier * IQR

        # Apply the filtering conditions
        if lower_bound is None:
            filtered_df = df_subset[df_subset[column] <= upper_bound]
        else:
            lower_bound = distance_value
            filtered_df = df_subset[(df_subset[column] <= upper_bound) & (df_subset[column] >= distance_value)]

        # Append filtered data for this group to the list
        filtered_dfs.append(filtered_df)

    # Concatenate all filtered groups
    filtered_df = pd.concat(filtered_dfs, ignore_index=True)

    # Calculate the number of removed rows and the percentage
    removed_count = df.shape[0] - filtered_df.shape[0]

    return filtered_df, removed_count