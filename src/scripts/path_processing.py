import numpy as np
import pandas as pd

def filter_finished_paths(finished_paths_df, pair_threshold=5, multiplier=1.5):
    """
    Downsample paths where the same start target pair has been played a lot more often than others, 
    and those wehre it has been played multiple times by the same player (same IpAddress and identifier).
    We also remove paths with distance 0 (start and target article are the same).
    Then filters paths based on the full path length for each distance group, using the IQR method to identify and remove outliers.
    
    Parameters:
    - finished_paths (pd.DataFrame): The input DataFrame containing path data.
    - pair_threshold (int): The maximum number of times the same start-target pair can be played. Default is 5.
    - multiplier (float): The multiplier for the IQR to determine the bounds. Default is 1.5.
    
    Returns:
    - filtered_finished_paths (pd.DataFrame): The filtered DataFrame without outliers.
    """

    filtered_dfs = []  # List to hold filtered data for each distance group

    # first downsample to 1 all paths where the start target pair has been played by the same player multtiple times 
    # (same IpAddress and identifier)
    finished_paths = finished_paths_df.groupby(['hashedIpAddress', 'identifier']).sample(n=1, random_state=42)

    # ------------------------------------

    # Downsample the paths so that the same start-target pair is not played more than the set threshold times

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Shuffle the DataFrame
    shuffled_df = finished_paths.sample(frac=1).reset_index(drop=True)

    # Group by 'identifier' and keep at most the first 3 rows for each group
    filtered_paths_sampled = (
        shuffled_df.groupby('identifier')
        .head(pair_threshold)
        .reset_index(drop=True)
    )

    finished_paths = filtered_paths_sampled

    # ------------------------------------

    # we noticed that there are some samples with distance 0 (start and target article are the same).
    # these also need to be filtered out
    finished_paths = finished_paths[~(finished_paths['distance']== 0)]

    # ------------------------------------

    # Apply the IQR method


    # Iterate over each unique distance
    for distance in finished_paths['distance'].unique():
        # Subset the DataFrame for the current distance group
        df_subset = finished_paths[finished_paths['distance'] == distance]

        # Compute IQR for the full path length
        Q1 = df_subset['full_path_length'].quantile(0.25)
        Q3 = df_subset['full_path_length'].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate lower and upper bounds based on IQR
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Filter rows within the bounds
        filtered_df = df_subset[(df_subset['full_path_length'] >= lower_bound) & (df_subset['full_path_length'] <= upper_bound)]

        # Append filtered data for this group to the list
        filtered_dfs.append(filtered_df)

    # Concatenate all filtered groups
    filtered_finished_paths = pd.concat(filtered_dfs, ignore_index=True)

    # Calculate the number of removed rows and the percentage
    removed_count = finished_paths_df.shape[0] - filtered_finished_paths.shape[0]
    removed_percentage = (removed_count / finished_paths_df.shape[0]) * 100

    # Print the summary
    print(f"In path length filtering a total of {removed_count} paths were removed from the finished paths, "
          f"which represents {removed_percentage:.3f}% of the original finished data.", 
          f" {filtered_finished_paths.shape[0]} paths remain.")

    return filtered_finished_paths



def filter_unfinished_paths(unfinished_paths_df, multiplier=1.5):
    """
    Remove all unfinished paths that the user did not actively restart (failure_reason != 'timeout').
    Filters unfinished paths based on the IQR method for each distance group. 
    The lower bound is the distance itself, and the upper bound is determined using the IQR method.

    Parameters:
    - unfinished_paths (pd.DataFrame): The input DataFrame containing unfinished path data.
    - multiplier (float): The multiplier for the IQR to determine the upper bound. Default is 1.5.

    Returns:
    - filtered_unfinished_paths (pd.DataFrame): The filtered DataFrame without outliers.
    - removed_count (int): The number of rows removed.
    - removed_percentage (float): The percentage of rows removed.
    """
    
    # First remove the paths that player did not actively fail (timeout)
    unfinished_paths = unfinished_paths_df[~(unfinished_paths_df['failure_reason'] == 'timeout')]

    filtered_dfs = []  # List to hold filtered data for each distance group

    # Iterate over each unique distance
    for distance_value in unfinished_paths['distance'].unique():
        # Subset the DataFrame for the current distance group
        df_subset = unfinished_paths[unfinished_paths['distance'] == distance_value]

        # Compute IQR for the full path length
        Q1 = df_subset['full_path_length'].quantile(0.25)
        Q3 = df_subset['full_path_length'].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate the upper bound based on IQR
        upper_bound = Q3 + multiplier * IQR

        # Apply the filtering conditions
        filtered_df = df_subset[
            (df_subset['full_path_length'] <= upper_bound) &  # Full path length <= upper bound
            (df_subset['simplified_path_length'] >= df_subset['distance'])  # Simplified path length >= distance
        ]

        # Append filtered data for this group to the list
        filtered_dfs.append(filtered_df)

    # Concatenate all filtered groups
    filtered_unfinished_paths = pd.concat(filtered_dfs, ignore_index=True)

    # Calculate the number of removed rows and the percentage
    removed_count = unfinished_paths_df.shape[0] - filtered_unfinished_paths.shape[0]
    removed_percentage = (removed_count / unfinished_paths_df.shape[0]) * 100

    # Print the summary
    print(f"A total of {removed_count} paths were removed from the unfinished paths, "
          f"which represents {removed_percentage:.3f}% of the original unfinished data.", 
          f" {filtered_unfinished_paths.shape[0]} paths remain.")

    return filtered_unfinished_paths



def filter_duration(df_in, pair_threshold=5, multiplier=1.5):
    """
    Filter the DataFrame based on the distance and duration bounds using the IQR method, 
    downsample to one IP address per identifier, and limit the number of plays per start-target pair.

    Parameters:
        df_in (pd.DataFrame): Input DataFrame with the following columns:
            - 'distance': Distance associated with the path
            - 'durationInSec': Duration associated with the path
        pair_threshold (int): The maximum number of times the same start-target pair can be played. Default is 5.

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_dfs = []  # List to hold filtered data for each distance group

    # Step 1: Downsample to one IP address per identifier
    df = df_in.groupby(['hashedIpAddress', 'identifier']).sample(n=1, random_state=42)

    # Step 2: Downsample to limit the number of plays per start-target pair
    # Shuffle the DataFrame for randomness
    np.random.seed(42)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # Group by 'identifier' and keep at most the first `pair_threshold` rows for each group
    df = (
        shuffled_df.groupby('identifier')
        .head(pair_threshold)
        .reset_index(drop=True)
    )

    # Step 3: We noticed that there are some samples with distance 0 (start and target article are the same).
    # these also need to be filtered out
    df = df[~(df['distance']== 0)]

    # Step 3: Filter by IQR for duration within each distance group
    for d in range(1, int(df['distance'].max()) + 1):
        # Filter the DataFrame for the current distance group
        df_d = df[df['distance'] == d]

        # Compute IQR for 'durationInSec'
        Q1 = df_d['durationInSec'].quantile(0.25)
        Q3 = df_d['durationInSec'].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate upper bound based on IQR
        upper_bound = Q3 + multiplier * IQR

        # Keep only rows within the upper bound
        filtered_df_d = df_d[df_d['durationInSec'] <= upper_bound]

        # Append filtered group to the list
        filtered_dfs.append(filtered_df_d)

    # Step 4: Concatenate all filtered groups
    filtered_df = pd.concat(filtered_dfs, ignore_index=True)

    # Step 5: Calculate the number of removed rows
    removed = df_in.shape[0] - filtered_df.shape[0]

    removed_percentage = (removed / df.shape[0]) * 100

    # Print the results 
    print(f"In path duration filtering a total of {removed} paths were removed from the finished paths, "
        f"which represents {removed_percentage:.3f}% of the original finished data.", 
        f" {filtered_df.shape[0]} paths remain.")
    
    return filtered_df