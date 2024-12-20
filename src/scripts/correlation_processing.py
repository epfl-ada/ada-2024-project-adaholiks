import pandas as pd
import matplotlib.pyplot as plt

def correlate_attributes(df, attributes, scores):
    """
    Computes correlations between attributes and composite scores directly.

    Parameters:
    - df (pd.DataFrame): DataFrame containing attributes and composite scores.
    - attributes (list of str): List of attribute column names.
    - scores (list of str): List of composite score column names.

    Returns:
    - dict: Dictionary containing correlation series for each composite score.
    """
    # Compute correlations
    correlations = {}
    for score in scores:
        corr = df[attributes].corrwith(df[score])
        correlations[score] = corr

    return correlations



import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


def load_and_preprocess_data(filepath):
    merged_df = feather.read_feather(filepath)
    #merged_df.set_index('article', inplace=True)
    features_df = merged_df.drop(columns=['composite_3', 'composite_2', 'PCA_composite_2'])
    targets = {
        'composite_3': merged_df[['composite_3']],
        'composite_2': merged_df[['composite_2']],
        'PCA_composite_2': merged_df[['PCA_composite_2']]
    }
    return features_df, targets


def split_and_standardize_data(features_df, target_df):
    train_features, test_features, train_targets, test_targets = train_test_split(
        features_df, target_df, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
     # Fit to continuous columns and transform
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # Convert back to DataFrame to retain column names
    train_scaled_df = pd.DataFrame(train_scaled, columns=train_features.columns, index=train_features.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=test_features.columns, index=test_features.index)

    # Add a constant column to the scaled DataFrames
    train_scaled_df = sm.add_constant(train_scaled_df)
    test_scaled_df = sm.add_constant(test_scaled_df)

    return train_features, test_features, train_targets, test_targets


def train_linear_regression(train_features, train_targets):
    train_features = sm.add_constant(train_features)
    model = sm.OLS(train_targets, train_features).fit()
    return model


def train_decision_tree(train_features, train_targets, test_features, test_targets):
    tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree.fit(train_features, train_targets)
    y_pred = tree.predict(test_features)
    mse = mean_squared_error(test_targets, y_pred)
    r2 = r2_score(test_targets, y_pred)

    # Create a Series with feature names as index and importance values as data
    feature_importances = pd.Series(tree.feature_importances_, index=train_features.columns)

    return tree, mse, r2, feature_importances

def plot_decision_tree(tree, feature_names):
    plt.figure(figsize=(15, 10))
    plot_tree(tree, feature_names=feature_names, filled=True, rounded=True)
    plt.show()








