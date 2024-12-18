
import pandas as pd
from src.utils.score_utils import *
from sklearn.decomposition import PCA

def compute_scores_df(length_filt_finished_paths, length_filt_paths, 
                             time_filt_paths, count_cutoff=30, scaling='standard', consider_start=True):
    """
    Computes composite scores by combining various metrics for articles 
    based on finished and filtered paths.
    
    Parameters:
    - length_filt_finished_paths (pd.DataFrame): DataFrame containing finished paths.
    - length_filt_paths (pd.DataFrame): DataFrame containing filtered paths.
    - count_cutoff (int): Minimum count threshold for calculations.
    - filter_duration_fn (function): Function to filter data based on duration (e.g., IQR filtering).
    - scaling (str): Scaling method to be used ('standard' or 'minmax'). Default is 'standard'.

    Returns:
    - pd.DataFrame: Composite DataFrame containing combined metrics.
    """
    # Calculate metrics
    print("Calculating click related scores...")
    avg_weight_df = calculate_avg_article_weights(length_filt_finished_paths, count_cutoff=count_cutoff, scaling=scaling, consider_start=consider_start)
    unfinished_ratio_df = calculate_unfinished_ratios(length_filt_paths, count_cutoff=count_cutoff, scaling=scaling, consider_start=consider_start)
    detour_ratio_df = calculate_detour_ratios(length_filt_finished_paths, count_cutoff=count_cutoff, scaling=scaling, consider_start=consider_start)

    sum_cweight_df = calculate_sum_article_cweights(length_filt_finished_paths, count_cutoff=count_cutoff, scaling=scaling, consider_start=consider_start)

    # Calculate speed metrics
    print("Calculating speed related scores...")
    #avg_speed_df = calc_avg_article_speed(time_filt_paths, count_cutoff=count_cutoff, scaling=scaling, consider_start=consider_start)
    #sum_cspeed_df = calc_sum_article_cspeed(time_filt_paths, count_cutoff=count_cutoff, scaling=scaling, consider_start=consider_start)
    adjusted_time_df = calc_avg_article_adjusted_time(time_filt_paths, count_cutoff=count_cutoff, scaling=scaling, consider_start=consider_start)
    sum_cadj_time_df = calc_sum_article_cadjusted_time(time_filt_paths, count_cutoff=count_cutoff, scaling=scaling, consider_start=consider_start)

    # Combine click metrics into a composite DataFrame
    composite_df = pd.DataFrame(index=avg_weight_df.index)
    composite_df['n_appearances'] = avg_weight_df['n_appearances']
    composite_df['avg_weight'] = avg_weight_df['weighted_avg']
    composite_df['weight_avg_scaled'] = avg_weight_df[scaling]

    composite_df['detour_ratio'] = detour_ratio_df['detour_ratio']
    composite_df['detour_ratio_scaled'] = detour_ratio_df[scaling]

    composite_df['unf_ratio'] = unfinished_ratio_df['unfinished_ratio']
    composite_df['unf_ratio_scaled'] = unfinished_ratio_df[scaling]

    composite_df['sum_cweight'] = sum_cweight_df['weighted_sum']
    composite_df['sum_cweight_scaled'] = sum_cweight_df[scaling]

    print(f"Number of unique articles in click score df: {composite_df.shape[0]}")

    # Speed metrics
    speed_df = pd.DataFrame(index=adjusted_time_df.index)
    speed_df['n_appearances'] = adjusted_time_df['n_appearances']
    #speed_df['avg_speed'] = avg_speed_df['avg_speed']
    #speed_df['avg_speed_scaled'] = avg_speed_df[scaling]

    #speed_df['sum_cspeed'] = sum_cspeed_df['sum_cspeed']
    #speed_df['sum_cspeed_scaled'] = sum_cspeed_df[scaling]

    speed_df['avg_adj_time'] = adjusted_time_df['avg_adj_time']
    speed_df['avg_adj_time_scaled'] = adjusted_time_df[scaling]

    speed_df['sum_cadj_time'] = sum_cadj_time_df['sum_cadj_time']
    speed_df['sum_cadj_time_scaled'] = sum_cadj_time_df[scaling]

    print(f"Number of unique articles in speed score df: {speed_df.shape[0]}")

    return composite_df.sort_values(by='avg_weight', ascending=False), speed_df.sort_values(by='avg_adj_time', ascending=True)


def calculate_composite_scores(composite_df):
    """
    Computes composite scores and PCA composite scores for quality and utility metrics.

    Parameters:
        composite_df (pd.DataFrame): DataFrame containing scaled features for score calculation.

    Returns:
        tuple: Two DataFrames (quality_scores_clicks, utility_scores_clicks) with calculated scores.
    """
    # Initialize DataFrames
    quality_scores_clicks = pd.DataFrame(index=composite_df.index)
    utility_scores_clicks = pd.DataFrame(index=composite_df.index)

    # Define weights for composite scores (quality)
    quality_weights_3 = {
        'weight_avg_scaled': 0.55,  # Example: 55% importance
        'unf_ratio_scaled': 0.2,   # Example: 20% importance
        'detour_ratio_scaled': 0.25  # Example: 25% importance
    }
    quality_weights_2 = {
        'weight_avg_scaled': 0.65,  
        'detour_ratio_scaled': 0.35
    }

    # Calculate quality scores
    quality_scores_clicks['n_appearances'] = composite_df['n_appearances']
    quality_scores_clicks['composite_3'] = (
        composite_df[['weight_avg_scaled', 'unf_ratio_scaled', 'detour_ratio_scaled']] * quality_weights_3
    ).sum(axis=1)
    quality_scores_clicks['composite_2'] = (
        composite_df[['weight_avg_scaled', 'detour_ratio_scaled']] * quality_weights_2
    ).sum(axis=1)

    # Compute composite score using PCA for only weight_avg_scaled and detour_ratio_scaled
    pca = PCA(n_components=1)
    quality_scores_clicks['PCA_composite_2'] = pca.fit_transform(
        composite_df[['weight_avg_scaled', 'detour_ratio_scaled']]
)

    # Define weights for composite scores (utility)
    utility_weights_3 = {
        'sum_cweight_scaled': 0.55,  
        'unf_ratio_scaled': 0.2,    
        'detour_ratio_scaled': 0.25
    }
    utility_weights_2 = {
        'sum_cweight_scaled': 0.65,
        'detour_ratio_scaled': 0.35
    }

    # Calculate utility scores
    utility_scores_clicks['n_appearances'] = composite_df['n_appearances']
    utility_scores_clicks['composite_3'] = (
        composite_df[['sum_cweight_scaled', 'unf_ratio_scaled', 'detour_ratio_scaled']] * utility_weights_3
    ).sum(axis=1)
    utility_scores_clicks['composite_2'] = (
        composite_df[['sum_cweight_scaled', 'detour_ratio_scaled']] * utility_weights_2
    ).sum(axis=1)

    # Return sorted DataFrames
    return (
        quality_scores_clicks.sort_values(by='composite_3', ascending=False),
        utility_scores_clicks.sort_values(by='composite_3', ascending=False)
)

def compute_binary_scores(quality_scores_clicks, utility_scores_clicks, time_scores, threshold=1):
    """
    Computes binary scores based on thresholding for quality and utility scores.

    Parameters:
        quality_scores_clicks (pd.DataFrame): DataFrame containing quality scores.
        utility_scores_clicks (pd.DataFrame): DataFrame containing utility scores.
        threshold (float): Threshold for binary classification. Default is 0.5.

    Returns:
        tuple: Two DataFrames (quality_binary, utility_binary) with binary scores.
    """
    quality_binary = binary_score(quality_scores_clicks, 'composite_3', threshold=threshold)
    utility_binary = binary_score(utility_scores_clicks, 'composite_3', threshold=threshold)

    avg_time_binary = binary_score(time_scores, 'avg_adj_time_scaled', threshold=threshold)
    csum_time_binary = binary_score(time_scores, 'sum_cadj_time_scaled', threshold=threshold)

    # Return binary DataFrames
    return quality_binary, utility_binary, avg_time_binary, csum_time_binary

