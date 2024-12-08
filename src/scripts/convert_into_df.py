import pandas as pd
from src.utils.load_utils import load_articles_into_df, load_plain_text
from src.utils.helpers import process_path, get_distance_between_articles

def articles_to_df(articles, categories, links, distances):
    """
    Combines multiple data sources to create an article dataframe.

    Parameters:
    - articles (pd.DataFrame): The base dataframe containing articles.
    - categories (pd.DataFrame): Dataframe with article categories.
    - links (pd.DataFrame): Dataframe containing article link mappings.
    - distances (dict): A dictionary where keys are article names and values are dicts of distances.

    Returns:
    - pd.DataFrame: The final merged article dataframe.
    """
    # Start with the articles dataframe
    article_dataframe = articles

    # Load unrendered unicode articles and add to dataframe
    articles_unrendered_unicode = load_articles_into_df(do_decode=False)
    article_dataframe['article_unrendered_unicode'] = articles_unrendered_unicode['article']

    # Merge with categories dataframe
    article_dataframe = pd.merge(article_dataframe, categories, on='article', how='left')

    # Merge with links dataframe
    article_dataframe = pd.merge(article_dataframe, links, left_on='article', right_on='linkSource', how='left')

    # Add distances column using the distances dictionary
    article_dataframe['distances'] = article_dataframe['article'].map(distances)

    # Load plain text for each article and add to dataframe
    article_dataframe['plain_text'] = article_dataframe['article_unrendered_unicode'].apply(load_plain_text)

    # Drop unnecessary columns
    article_dataframe.drop(columns=['article_unrendered_unicode', 'linkSource',], inplace=True)

    return article_dataframe


def paths_to_df(paths_finished, paths_unfinished, article_df):
    """
    Creates a paths dataframe by combining finished and unfinished paths and processes additional features.

    Parameters:
    - paths_finished (pd.DataFrame): Dataframe containing finished paths.
    - paths_unfinished (pd.DataFrame): Dataframe containing unfinished paths.
    - article_df (pd.DataFrame): Dataframe containing article information.

    Returns:
    - pd.DataFrame: The final processed paths dataframe.
    """
    # Add "finished" column to paths_finished
    paths_finished['finished'] = True
    paths_finished['failure_reason'] = None

    # Extract the start_article and target_article from the `path` column
    paths_finished['start_article'] = paths_finished['path'].apply(lambda x: x.split(';')[0])
    paths_finished['target_article'] = paths_finished['path'].apply(lambda x: x.split(';')[-1])

    # Add "finished" column to paths_unfinished
    paths_unfinished['finished'] = False

    # Clean the paths_unfinished dataframe
    # Rename 'type' to 'failure_reason'
    paths_unfinished.rename(columns={'type': 'failure_reason', 'target': 'target_article'}, inplace=True)

    # Extract the start_article from the `path` column
    paths_unfinished['start_article'] = paths_unfinished['path'].apply(lambda x: x.split(';')[0])

    # Combine finished and unfinished paths
    paths_df = pd.concat([paths_finished, paths_unfinished])

    # Reset index to have unique indexes
    paths_df = paths_df.reset_index(drop=True)

    # Add an identifier for the same start-target articles pairs
    paths_df['identifier'] = paths_df.groupby(['start_article', 'target_article']).ngroup()

    # Add two new columns to the paths dataset with processed paths
    paths_df[['full_path', 'simplified_path']] = paths_df['path'].apply(lambda x: pd.Series(process_path(x)))

    # Add a column for distances between start and target articles
    paths_df['distance'] = paths_df.apply(
        lambda row: get_distance_between_articles(article_df, row['start_article'], row['target_article']), axis=1
    )

    # Add columns for full and simplified path lengths
    paths_df['full_path_length'] = paths_df['full_path'].apply(lambda x: len(x) - 1)  # n-1 is the distance
    paths_df['simplified_path_length'] = paths_df['simplified_path'].apply(lambda x: len(x) - 1)

    return paths_df
