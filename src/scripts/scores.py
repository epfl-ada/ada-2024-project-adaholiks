
import pandas as pd
from src.utils.score_utils import *

def compute_scores_df(length_filt_finished_paths, length_filt_paths, 
                             time_filt_paths, count_cutoff=30, scaling='standard'):
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
    avg_weight_df = calculate_avg_article_weights(length_filt_finished_paths, count_cutoff=count_cutoff, scaling=scaling)
    unfinished_ratio_df = calculate_unfinished_ratios(length_filt_paths, count_cutoff=count_cutoff, scaling=scaling)
    detour_ratio_df = calculate_detour_ratios(length_filt_finished_paths, count_cutoff=count_cutoff, scaling=scaling)

    sum_cweight_df = calculate_sum_article_cweights(length_filt_finished_paths, count_cutoff=count_cutoff, scaling=scaling)

    # Calculate speed metrics
    print("Calculating speed related scores...")
    avg_speed_df = calc_avg_article_speed(time_filt_paths, count_cutoff=count_cutoff, scaling=scaling)
    sum_cspeed_df = calc_sum_article_cspeed(time_filt_paths, count_cutoff=count_cutoff, scaling=scaling)

    # Combine metrics into a composite DataFrame
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

    composite_df['avg_speed'] = avg_speed_df['avg_speed']
    composite_df['avg_speed_scaled'] = avg_speed_df[scaling]

    composite_df['sum_cspeed'] = sum_cspeed_df['sum_cspeed']
    composite_df['sum_cspeed_scaled'] = sum_cspeed_df[scaling]

    print(f"Number of unique articles: {composite_df.shape[0]}")
    return composite_df.sort_values(by='avg_weight', ascending=False)

# Example usage:
# composite_df = compute_composite_scores(finished_paths, fi
