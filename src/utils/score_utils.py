import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def calculate_avg_article_weights(df, count_cutoff=30, scaling=None):
    """
    Calculate the average weights of articles from a DataFrame containing path information.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following columns:
            - 'simplified_path': List of articles in the path
            - 'simplified_path_length': Length of the simplified path
            - 'distance': Distance associated with the path
        scaling (str): Type of scaling to use. Options are 'minmax', 'standard', or None
        count_cutoff (int): Minimum number of appearances for an article to be considered

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'article': Article name
            - 'n_appearances': Number of times the article appeared in paths
            - 'weighted_avg': Weighted average of distances for the article
    """
    # Copy and preprocess the DataFrame
    df = df[['simplified_path', 'simplified_path_length', 'distance']].copy()
    df['simplified_path'] = df['simplified_path'].apply(lambda l: l[1:-1])  # Remove start and end articles

    # Calculate weight for each path
    df['weight'] = df['distance'] / df['simplified_path_length']

    # Initialize an empty DataFrame to store results
    avg_article_weight_df = pd.DataFrame(columns=['article', 'n_appearances', 'weighted_avg'])
    avg_article_weight_df.set_index('article', inplace=True)

    # Iterate through each row to calculate weights
    for _, row in df.iterrows():
        weight = row['weight']
        simplified_path = row['simplified_path']

        for article in simplified_path:
            if article not in avg_article_weight_df.index:
                avg_article_weight_df.loc[article] = [0, 0.0]

            # Update counts and weighted sums
            avg_article_weight_df.at[article, 'n_appearances'] += 1
            avg_article_weight_df.at[article, 'weighted_avg'] += weight

    # Calculate the weighted average by dividing weighted sum by counts
    avg_article_weight_df['weighted_avg'] = avg_article_weight_df['weighted_avg'] / avg_article_weight_df['n_appearances']

    # Filter out articles that appear less than the cutoff
    avg_article_weight_df = avg_article_weight_df[avg_article_weight_df['n_appearances'] >= count_cutoff]

    # Normalize the weighted average
    if scaling is not None:

        if scaling == 'minmax':
            scaler = MinMaxScaler()
        elif scaling == 'standard':
            scaler = StandardScaler()

        avg_article_weight_df[scaling] = scaler.fit_transform(avg_article_weight_df[['weighted_avg']])


    #print(f"Number of unique articles after weighting: {avg_article_weight_df.shape[0]}")

    return avg_article_weight_df#.reset_index()


# ------------------------------------------------

# function for utils later to get the average weights of articles from a DataFrame containing path information

def calculate_sum_article_cweights(df, count_cutoff=30, scaling=None):
    """
    Calculate the sum of the centered weights of articles from a DataFrame containing path information.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following columns:
            - 'simplified_path': List of articles in the path
            - 'simplified_path_length': Length of the simplified path
            - 'distance': Distance associated with the path
        scaling (str): Type of scaling to use. Options are 'minmax', 'standard', or None
        count_cutoff (int): Minimum number of appearances for an article to be considered

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'article': Article name
            - 'n_appearances': Number of times the article appeared in paths
            - 'weighted_sum': sum of the centered path weights for the article
    """
    # Copy and preprocess the DataFrame
    df = df[['simplified_path', 'simplified_path_length', 'distance']].copy()
    df['simplified_path'] = df['simplified_path'].apply(lambda l: l[1:-1])  # Remove start and end articles

    # Calculate weight for each path
    df['weight'] = df['distance'] / df['simplified_path_length']

    # Calculate mean weight
    article_mean_weight = (df['weight'] * (df['simplified_path_length']-1)).sum() / (df['simplified_path_length']-1).sum() # -1 beacuse we don't want to include the target article

    # Center the weights by subtracting the mean
    df['centered_weight'] = df['weight'] - article_mean_weight

    # Initialize an empty DataFrame to store results
    sum_article_cweight_df = pd.DataFrame(columns=['article', 'n_appearances', 'weighted_sum'])
    sum_article_cweight_df.set_index('article', inplace=True)

    # Iterate through each row to calculate weights
    for _, row in df.iterrows():
        cweight = row['centered_weight']
        simplified_path = row['simplified_path']

        for article in simplified_path:
            if article not in sum_article_cweight_df.index:
                sum_article_cweight_df.loc[article] = [0, 0.0]

            # Update counts and weighted sums
            sum_article_cweight_df.at[article, 'n_appearances'] += 1
            sum_article_cweight_df.at[article, 'weighted_sum'] += cweight

    # Filter out articles that appear less than the cutoff
    sum_article_cweight_df = sum_article_cweight_df[sum_article_cweight_df['n_appearances'] >= count_cutoff]

    # Normalize the weighted average
    if scaling is not None:

        if scaling == 'minmax':
            scaler = MinMaxScaler()
        elif scaling == 'standard':
            scaler = StandardScaler()

        sum_article_cweight_df[scaling] = scaler.fit_transform(sum_article_cweight_df[['weighted_sum']])


    #print(f"Number of unique articles after weighting: {sum_article_cweight_df.shape[0]}")

    return sum_article_cweight_df


# ------------------------------------------------


# code a function that returns the ratio of the number of times an article appears in unfinished paths over the total number of times it appears

def calculate_unfinished_ratios(in_df, count_cutoff=30, scaling=None):
    """
    Calculate the ratio of the number of times an article appears in unfinished paths over the total number of times it appears.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following columns:
            - 'simplified_path': List of articles in the path
        count_cutoff (int): Minimum number of appearances for an article to be considered
        scaling (str): Type of scaling to use. Options are 'minmax', 'standard', and 'robust' or None

    Returns:
        pd.Series: A Series containing the ratio for each article
    """
    # Copy and preprocess the DataFrame
    df = in_df[['simplified_path', 'finished']].copy()
    df['simplified_path'] = df['simplified_path'].apply(lambda l: l[1:-1])  # Remove start and end articles

    # Initialize a dictionary to store counts
    article_counts = {}
    unfinished_counts = {}

    # Iterate through each row to calculate counts
    for _, row in df.iterrows():
        simplified_path = row['simplified_path']
        finished = row['finished']

        for article in simplified_path:
            article_counts[article] = article_counts.get(article, 0) + 1
        
        if not finished:
            for article in simplified_path:
                unfinished_counts[article] = unfinished_counts.get(article, 0) + 1

    # Convert the dictionary to a Series
    article_counts = pd.Series(article_counts)
    unfinished_counts = pd.Series(unfinished_counts)

    ratio = unfinished_counts / article_counts

    ratio_df = pd.DataFrame({
    'n_appearances': article_counts,
    'unfinished_counts': unfinished_counts,
    'unfinished_ratio': ratio
    }).fillna(0)

    # cut off
    ratio_df = ratio_df[ratio_df['n_appearances'] >= count_cutoff]

    # scaling
    if scaling is not None:
        if scaling == 'minmax':
            scaler = MinMaxScaler()
            ratio_df[scaling] = scaler.fit_transform(1-ratio_df[['unfinished_ratio']]) # want the finished ratio, so bigger is better
        elif scaling == 'standard':
            scaler = StandardScaler()
            ratio_df[scaling] = -scaler.fit_transform(ratio_df[['unfinished_ratio']]) # minus sign so bigger is better

    #print(f"Number of unique articles: {len(article_counts)}")
    print(f"Ratio of unfinished over finished paths: {1-df['finished'].mean()}")
    return ratio_df


# ------------------------------------------------


# code a function that counts the number of dead ends an article has (difference between full path list content and simplified path list content)

def calculate_detour_ratios(in_df, count_cutoff=1, scaling=None):
    """
    Calculate the detour ratio for articles based on the full path and simplified path.

    Parameters:
        in_df (pd.DataFrame): Input DataFrame with the following columns:
            - 'full_path': List of articles in the full path
            - 'simplified_path': List of articles in the simplified path
        count_cutoff (int): Minimum number of detours for an article to be considered.
        scaling (str): Type of scaling to use. Options are 'minmax', 'standard', and 'robust' or None.

    Returns:
        pd.DataFrame: A DataFrame containing the detour ratio and scaled values for each article.
    """
    # Copy and preprocess the DataFrame
    df = in_df[['full_path', 'simplified_path']].copy()
    df['simplified_path'] = df['simplified_path'].apply(lambda l: l[1:-1])  # Remove start and end articles
    df['full_path'] = df['full_path'].apply(lambda l: l[1:-1])  # Remove start and end articles

    # Initialize dictionaries to store counts
    detour_counts = {}
    total_counts = {}

    # Iterate through each row to calculate detour counts and total appearances
    for _, row in df.iterrows():
        full_path = row['full_path']
        simplified_path = row['simplified_path']

        # Count total appearances for articles in the full path
        for article in full_path:
            total_counts[article] = total_counts.get(article, 0) + 1

        # Find detour articles by subtracting the simplified path from the full path
        detour_articles = set(full_path) - set(simplified_path)
        for article in detour_articles:
            detour_counts[article] = detour_counts.get(article, 0) + 1

    # Convert counts to Series
    detour_counts = pd.Series(detour_counts)
    total_counts = pd.Series(total_counts)

    # Fill missing detour counts with 0 for articles with no detours
    detour_counts = detour_counts.reindex(total_counts.index, fill_value=0)

    # Calculate detour ratio
    detour_ratios = detour_counts / total_counts

    # Create a DataFrame with detour counts and ratios
    detour_df = pd.DataFrame({
        'detour_count': detour_counts,
        'total_count': total_counts,
        'detour_ratio': detour_ratios
    }).loc[detour_ratios.index]

    # Filter out articles with detour ratio less than the count_cutoff
    detour_df = detour_df[detour_df['total_count'] >= count_cutoff]

    if scaling is not None:
        # normalize
        if scaling == 'minmax':
            scaler = MinMaxScaler()
            detour_df[scaling] = scaler.fit_transform(1-detour_df[['detour_ratio']])
        elif scaling == 'standard':
            scaler = StandardScaler()
            detour_df[scaling] = -scaler.fit_transform(detour_df[['detour_ratio']])

    #print(f"Number of unique articles after detour ratio calculation: {len(detour_df)}")

    return detour_df



# ------------------------------------------------

def calc_avg_article_time(df, count_cutoff=30, scaling=None):
    """
    Calculate the average speed of articles from a DataFrame containing path information.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following columns:
            - 'simplified_path': List of articles in the path
            - 'durationInSec': Duration associated with the path
        count_cutoff (int): Minimum number of appearances for an article to be considered
        scaling (str): Type of scaling to use. Options are 'minmax', 'standard', or None.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'article': Article name
            - 'n_appearances': Number of times the article appeared in paths
            - 'avg_speed': Average speed of the article
    """
    # Copy and preprocess the DataFrame
    df = df[['simplified_path', 'durationInSec']].copy()

    df['simplified_path'] = df['simplified_path'].apply(lambda l: l[1:-1])  # Remove start and end articles

    # Initialize an empty DataFrame to store results
    avg_article_speed_df = pd.DataFrame(columns=['article', 'n_appearances', 'avg_speed'])
    avg_article_speed_df.set_index('article', inplace=True)

    # Iterate through each row to calculate speeds
    for _, row in df.iterrows():
        speed = row['durationInSec']
        simplified_path = row['simplified_path']

        for article in simplified_path:
            if article not in avg_article_speed_df.index:
                avg_article_speed_df.loc[article] = [0, 0.0]

            # Update counts and sums
            avg_article_speed_df.at[article, 'n_appearances'] += 1
            avg_article_speed_df.at[article, 'avg_speed'] += speed

    # Calculate the average speed by dividing sum by counts
    avg_article_speed_df['avg_speed'] = avg_article_speed_df['avg_speed'] / avg_article_speed_df['n_appearances']

    # Filter out articles that appear less than the cutoff
    avg_article_speed_df = avg_article_speed_df[avg_article_speed_df['n_appearances'] >= count_cutoff]

    if scaling is not None:
        # Normalize the average speed
        if scaling == 'minmax':
            scaler = MinMaxScaler()
            avg_article_speed_df[scaling] = scaler.fit_transform(1-avg_article_speed_df[['avg_speed']])
        elif scaling == 'standard':
            scaler = StandardScaler()
            avg_article_speed_df[scaling] = -scaler.fit_transform(avg_article_speed_df[['avg_speed']])

    #print(f"Number of unique articles after time calc: {avg_article_speed_df.shape[0]}")

    return avg_article_speed_df#.reset_index()

def calc_avg_article_speed(df, count_cutoff=30, scaling=None):
    """
    Calculate the average speed and average path length of articles 
    from a DataFrame containing path information.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following columns:
            - 'simplified_path': List of articles in the path
            - 'durationInSec': Duration associated with the path
            - 'full_path_length': Total length of the path
        count_cutoff (int): Minimum number of appearances for an article to be considered.
        scaling (str): Type of scaling to use. Options are 'minmax', 'standard', or None.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'article': Article name.
            - 'n_appearances': Number of times the article appeared in paths.
            - 'avg_speed': Average speed of the article.
            - 'avg_path_length': Average full_path_length of the article.
    """
    # Copy and preprocess the DataFrame
    df = df[['simplified_path', 'durationInSec', 'full_path_length']].copy()

    # Calculate the speed for each path
    df['speed'] = df['full_path_length'] / df['durationInSec']

    # Remove the start and end articles from the simplified path
    df['simplified_path'] = df['simplified_path'].apply(lambda l: l[1:-1])  # Adjust as per your input structure

    # Initialize an empty DataFrame to store results
    avg_article_speed_df = pd.DataFrame(columns=['article', 'n_appearances', 'avg_speed', 'avg_path_length'])
    avg_article_speed_df.set_index('article', inplace=True)

    # Iterate through each row to calculate speeds and path lengths
    for _, row in df.iterrows():
        speed = row['speed']
        path_length = row['full_path_length']
        simplified_path = row['simplified_path']

        for article in simplified_path:
            if article not in avg_article_speed_df.index:
                avg_article_speed_df.loc[article] = [0, 0.0, 0.0]

            # Update counts, speed sums, and path length sums
            avg_article_speed_df.at[article, 'n_appearances'] += 1
            avg_article_speed_df.at[article, 'avg_speed'] += speed
            avg_article_speed_df.at[article, 'avg_path_length'] += path_length

    # Calculate the averages
    avg_article_speed_df['avg_speed'] = avg_article_speed_df['avg_speed'] / avg_article_speed_df['n_appearances']
    avg_article_speed_df['avg_path_length'] = avg_article_speed_df['avg_path_length'] / avg_article_speed_df['n_appearances']

    # Filter out articles that appear less than the cutoff
    avg_article_speed_df = avg_article_speed_df[avg_article_speed_df['n_appearances'] >= count_cutoff]

    if scaling is not None:
        # Normalize the average speed
        if scaling == 'minmax':
            scaler = MinMaxScaler()
            avg_article_speed_df[scaling] = scaler.fit_transform(avg_article_speed_df[['avg_speed']])
        elif scaling == 'standard':
            scaler = StandardScaler()
            avg_article_speed_df[scaling] = scaler.fit_transform(avg_article_speed_df[['avg_speed']])

    #print(f"Number of unique articles after speed and path length calc: {avg_article_speed_df.shape[0]}")

    return avg_article_speed_df  # Return the updated DataFrame


# ------------------------------------------------

def calc_sum_article_cspeed(df, count_cutoff=30, scaling=None):
    """
    Calculate the sum of the centered speeds of articles from a DataFrame containing path information.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following columns:
            - 'simplified_path': List of articles in the path
            - 'durationInSec': Duration associated with the path
            - 'full_path_length': Total length of the path
        count_cutoff (int): Minimum number of appearances for an article to be considered.
        scaling (str): Type of scaling to use. Options are 'minmax', 'standard', or None.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'article': Article name.
            - 'n_appearances': Number of times the article appeared in paths.
            - 'sum_cspeed': Sum of the centered path speeds for the article.
    """
    # Copy and preprocess the DataFrame
    df = df[['simplified_path', 'durationInSec', 'full_path_length']].copy()

    # Calculate the speed for each path
    df['speed'] = df['full_path_length'] / df['durationInSec']

    # Calculate the mean speed
    article_mean_speed = (df['speed'] * (df['full_path_length']-1)).sum() / (df['full_path_length']-1).sum()

    # Center the speeds by subtracting the mean
    df['centered_speed'] = df['speed'] - article_mean_speed

    # Remove the start and end articles from the simplified path
    df['simplified_path'] = df['simplified_path'].apply(lambda l: l[1:-1])  # Adjust as per your input structure

    # Initialize an empty DataFrame to store results
    sum_cspeed_df = pd.DataFrame(columns=['article', 'n_appearances', 'sum_cspeed'])
    sum_cspeed_df.set_index('article', inplace=True)

    # Iterate through each row to calculate speeds
    for _, row in df.iterrows():
        cspeed = row['centered_speed']
        simplified_path = row['simplified_path']

        for article in simplified_path:
            if article not in sum_cspeed_df.index:
                sum_cspeed_df.loc[article] = [0, 0.0]

            # Update counts and sums
            sum_cspeed_df.at[article, 'n_appearances'] += 1
            sum_cspeed_df.at[article, 'sum_cspeed'] += cspeed

    # Filter out articles that appear less than the cutoff
    sum_cspeed_df = sum_cspeed_df[sum_cspeed_df['n_appearances'] >= count_cutoff]

    if scaling is not None:
        # Normalize the sum of centered speeds
        if scaling == 'minmax':
            scaler = MinMaxScaler()
            sum_cspeed_df[scaling] = scaler.fit_transform(sum_cspeed_df[['sum_cspeed']])
        elif scaling == 'standard':
            scaler = StandardScaler()
            sum_cspeed_df[scaling] = scaler.fit_transform(sum_cspeed_df[['sum_cspeed']])
    
    #print(f"Number of unique articles after speed calc: {sum_cspeed_df.shape[0]}")

    return sum_cspeed_df  # Return the updated DataFrame
