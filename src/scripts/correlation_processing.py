import pandas as pd
import matplotlib.pyplot as plt

def rank_attributes(df, attributes, scores):
    """
    Ranks attributes and computes correlations with composite scores.

    Parameters:
    - df (pd.DataFrame): DataFrame containing attributes and composite scores
    - attributes (list of str): List of metric column names
    - composites (list of str): List of composite score column names

    Returns:
    - dict: Dictionary containing correlation series for each composite score
    """
    ranked_df = df.copy()

    # Rank each attribute
    for attribute in attributes:
        ascending = True if attribute == 'average_cosine_distance' else False
        ranked_df[f'rank_{attribute}'] = ranked_df[attribute].rank(ascending=ascending)

    # rank each score
    for score in scores:
        ranked_df[f'rank_{score}'] = ranked_df[score].rank(ascending=False)

    # Compute correlations
    correlations = {}
    for score in scores:
        corr = ranked_df[[f'rank_{attribute}' for attribute in attributes]].corrwith(ranked_df[f'rank_{score}'])
        correlations[score] = corr

    return correlations

