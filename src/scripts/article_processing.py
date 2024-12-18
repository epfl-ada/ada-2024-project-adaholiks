import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import re

def add_incoming_links(article_df):
    """
    Adds the number of incoming links as a new attribute to the DataFrame.
    """
    all_incoming_links = article_df['linkTarget'].explode()
    incoming_link_counts = all_incoming_links.value_counts()
    article_df['incoming_links'] = article_df['article'].map(incoming_link_counts).fillna(0).astype(int)
    return article_df

def add_num_hyperlinks(article_df):
    """
    Adds the number of outgoing hyperlinks as a new attribute to the DataFrame.
    """
    article_df['linkTarget'] = article_df['linkTarget'].apply(
        lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
    )
    article_df['num_hyperlinks'] = article_df['linkTarget'].apply(len)
    return article_df

def add_length_and_hyperlink_density(article_df):
    """
    Adds the number of characters and hyperlink density (num_hyperlinks / num_characters) as a new attribute to the DataFrame.
    """
    article_df['plain_text'] = article_df['plain_text'].fillna("").astype(str)
    article_df['num_characters'] = article_df['plain_text'].apply(len)
    article_df['hyperlink_density'] = article_df.apply(
        lambda row: row['num_hyperlinks'] / row['num_characters'] if row['num_characters'] > 0 else 0,
        axis=1
    )
    return article_df

def add_average_cosine_distance(article_df, embeddings):

    """
    Adds the average cosine distance from each article embedding to the embedding of it's outgoing links.

    Parameters:
    - article_df: DataFrame containing the articles
    - embeddings: A dictionary or DataFrame mapping article titles to their embeddings

    Returns:
    - Updated article_df with 'average_cosine_distance' column
    """

    # Convert the DataFrame into a single column of arrays
    embeddings['embeddings'] = embeddings.apply(lambda row: row.tolist(), axis=1)

    # Create a new DataFrame with a single column of arrays
    single_column_embeddings = embeddings[['embeddings']]

    single_column_embeddings.rename(columns={'array': 'embeddings'}, inplace=True)

    # attach the embeddings row wise
    article_df = pd.concat([article_df, single_column_embeddings], axis=1)

    for article_name in article_df['article']:
        # Filter the current article
        article = article_df[article_df['article'] == article_name]
        cosine_distances = []

        # Get outgoing articles linked from the current article
        outgoing_articles = article['linkTarget']

        for outgoing_article_row in outgoing_articles:
            for outgoing_article_name in outgoing_article_row:
                # Find the outgoing article
                outgoing_article = article_df[article_df['article'] == outgoing_article_name]

                # Extract embeddings
                article_embedding = np.array(article['embeddings'].iloc[0])  # Extract as numeric array
                outgoing_embedding = np.array(outgoing_article['embeddings'].iloc[0])  # Same for outgoing article

                # Compute cosine distance
                cosine_distance = cosine(article_embedding, outgoing_embedding)
                cosine_distances.append(cosine_distance)

        # Compute the average cosine distance
        cosine_distances = np.array(cosine_distances)
        average_cosine_distance = np.mean(cosine_distances)

        # Store the value in the DataFrame
        article_df.loc[article_df['article'] == article_name, 'average_cosine_distance'] = average_cosine_distance

    return article_df

def add_vocabulary_richness(article_df):
    """
    Adds the vocabulary richness attribute to the DataFrame.

    Parameters:
    - article_df: DataFrame containing a 'plain_text' column with the article text.

    Returns:
    - Updated article_df with 'vocabulary_richness' column.
    """
    def compute_vocabulary_richness(text):
        # Tokenize the text (split into words)
        words = re.findall(r'\b\w+\b', text.lower())  # Lowercase and split on words
        total_words = len(words)
        unique_words = len(set(words))
        if total_words == 0:  # Avoid division by zero
            return 0
        return unique_words / total_words

    # Apply the vocabulary richness function to the plain_text column
    article_df['vocabulary_richness'] = article_df['plain_text'].apply(compute_vocabulary_richness)
    return article_df
