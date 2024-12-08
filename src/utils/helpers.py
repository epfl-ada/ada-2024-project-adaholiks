import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------------------------------------------
# ------------------------ helper functions for converting the data into dataframes ------------------------
def process_path(path_string):
    # Split the input string by the ";" delimiter
    links = path_string.split(";")
    
    # Initialize lists for full path and simplified path
    full_path = []
    simplified_path = []
    
    # Traverse through each link in the path
    for link in links:
        # Handle backward steps
        if link == "<":
            if simplified_path: # make sure list is not empty befor pop
                simplified_path.pop()  # Go back by removing last entry
        else:
            full_path.append(link)
            simplified_path.append(link)
    
    return full_path, simplified_path


# a function that takes as input two articles and computes the distance between them using the distance col from article_df 

def get_distance_between_articles(df, article1, article2):
    """
    Compute the distance between two articles based on the distance data in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing articles and their distance dictionaries.
        article1 (str): The name of the first article.
        article2 (str): The name of the second article.
        
    Returns:
        int or None: The distance between article1 and article2. Returns None if not found.
    """
    # Find the row where the article is article1
    row_article1 = df[df['article'] == article1]
    
    if row_article1.empty:
        print(f"Article '{article1}' not found in the DataFrame.")
        return None
    
    # Extract the dictionary of distances for article1
    distances_dict = row_article1['distances'].values[0]
    
    # Retrieve the distance to article2
    distance = distances_dict.get(article2)

    # this will return either an int or None
    return distance

# --------------------------------------------------------------------------------------------
# ------------------------ Plotting functions for path processing-----------------------------

def plot_path_length_distributions(df, full_path_col='full_path_length', simplified_path_col='simplified_path_length'):
    """
    Creates side-by-side bar plots for the distribution of path lengths (full paths and simplified paths).

    Parameters:
        df (pd.DataFrame): The input DataFrame containing path length data.
        full_path_col (str): Column name for full path lengths. Default is 'full_path_length'.
        simplified_path_col (str): Column name for simplified path lengths. Default is 'simplified_path_length'.
    """
    # Count occurrences of each path length for the specified columns
    len_full_counts = df[full_path_col].value_counts().sort_index()
    len_simplified_counts = df[simplified_path_col].value_counts().sort_index()

    # Create a figure with 2 subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the bar plot for full path lengths on the first subplot
    len_full_counts.plot(kind='bar', color='#3498db', edgecolor='black', alpha=0.75, ax=axes[0])
    axes[0].set_title("Bar Plot of Finished Full Paths", fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
    axes[0].set_xlabel("Number of Visited Articles (number of clicks)", fontsize=14, color='#34495E', labelpad=15)
    axes[0].set_ylabel("Frequency", fontsize=14, color='#34495E', labelpad=15)
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='#BDC3C7')  # Subtle gridlines

    # Plot the bar plot for simplified path lengths on the second subplot
    len_simplified_counts.plot(kind='bar', color='#e74c3c', edgecolor='black', alpha=0.75, ax=axes[1])
    axes[1].set_title("Bar Plot of Finished Simplified Paths", fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
    axes[1].set_xlabel("Direct Path Length (no back clicks)", fontsize=14, color='#34495E', labelpad=15)
    axes[1].set_ylabel("Frequency", fontsize=14, color='#34495E', labelpad=15)
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='#BDC3C7')  # Subtle gridlines

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_duration_histogram(df, duration_col='durationInSec', bins=30):
    """
    Creates a histogram for the duration column in the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing duration data.
        duration_col (str): Column name for the duration data. Default is 'durationInSec'.
        bins (int): Number of bins for the histogram. Default is 30.
    """
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[duration_col], bins=bins, color='#9b59b6', edgecolor='black', alpha=0.75)

    # Add titles and labels
    plt.title("Histogram of Duration (in Seconds)", fontsize=18, fontweight='bold', color='#2C3E50', pad=20)
    plt.xlabel("Duration (Seconds)", fontsize=14, color='#34495E', labelpad=15)
    plt.ylabel("Frequency", fontsize=14, color='#34495E', labelpad=15)

    # Add gridlines
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='#BDC3C7')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_by_distance(df, plot_col, distance_col='distance', x_label="Value", plot_type='bar', bins=None):
    """
    Creates plots (bar or histogram) for the distribution of a specified column, grouped by distance values.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        plot_col (str): The column name to plot.
        distance_col (str): Column name for distance groups. Default is 'distance'.
        x_label (str): Label for the x-axis of the subplots.
        plot_type (str): Type of plot to create ('bar' or 'hist'). Default is 'bar'.
        bins (int or sequence): Number of bins for histogram plots. Optional.
    """
    if plot_type not in ['bar', 'hist']:
        raise ValueError("plot_type must be either 'bar' or 'hist'")
    
    # Get unique distance values sorted
    distances = sorted(df[distance_col].unique())

    # Create a figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for idx, distance in enumerate(distances):
        # Filter the DataFrame for the current distance
        df_distance = df[df[distance_col] == distance]
        
        if plot_type == 'bar':
            # Count occurrences of each value in the specified column
            value_counts = df_distance[plot_col].value_counts().sort_index()

            # Bar plot on the corresponding subplot
            value_counts.plot(kind='bar', color='#2ecc71', edgecolor='black', alpha=0.75, ax=axes[idx])
        elif plot_type == 'hist':
            # Histogram plot
            axes[idx].hist(df_distance[plot_col], bins=bins, color='#2ecc71', edgecolor='black', alpha=0.75)
        
        # Set titles and labels
        axes[idx].set_title(f"Distance {distance}", fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
        axes[idx].set_xlabel(x_label, fontsize=14, color='#34495E', labelpad=15)
        axes[idx].set_ylabel("Frequency", fontsize=14, color='#34495E', labelpad=15)
        axes[idx].grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='#BDC3C7')  # Subtle gridlines

    # Hide any unused subplots if distances are fewer than 6
    for idx in range(len(distances), len(axes)):
        axes[idx].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_styled_bar_chart(
    data, 
    x_label="X-Axis Label", 
    y_label="Y-Axis Label", 
    title="Bar Chart Title", 
    figsize=(8, 6), 
    rotation=90, 
    bar_color='#3498db', 
    grid_color='#BDC3C7'
):
    """
    Plots a styled bar chart for the given data.

    Parameters:
        data (pd.Series): The data to plot, with the index as categories and values as frequencies.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the chart.
        figsize (tuple): Figure size (width, height). Default is (8, 6).
        rotation (int): Rotation angle for the x-axis tick labels. Default is 90.
        bar_color (str): Color for the bars. Default is '#3498db'.
        grid_color (str): Color for the grid lines. Default is '#BDC3C7'.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    data.plot(kind='bar', color=bar_color, edgecolor='black', alpha=0.75, ax=ax)

    # Customize title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
    ax.set_xlabel(x_label, fontsize=14, color='#34495E', labelpad=15)
    ax.set_ylabel(y_label, fontsize=14, color='#34495E', labelpad=15)

    # Add grid lines
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color=grid_color)

    # Rotate x-tick labels
    ax.tick_params(axis='x', rotation=rotation)

    # Show the plot
    plt.show()

# ----------------------------------------------------------------------------------------------------------
# --------------------------- Plotting function for the scores ---------------------------------------------


def plot_scaled_score_correlations(composite_df, 
                                   scaled_columns = ['weight_avg_scaled', 
                                                     'unf_ratio_scaled', 
                                                     'detour_ratio_scaled', 
                                                     'sum_cweight_scaled', 
                                                     'avg_speed_scaled', 
                                                     'sum_cspeed_scaled'
]):
    """
    Generates a pairplot to visualize the correlations among scaled score columns in a DataFrame.
    
    Parameters:
        composite_df (pd.DataFrame): The input DataFrame containing scaled score columns.
        scaled_columns (list): List of column names to consider for the correlation plot.
    
    Returns:
        sns.PairGrid: The generated Seaborn PairGrid object.
    """
    # Filter the DataFrame to include only the specified scaled columns
    composite_df_scaled = composite_df[scaled_columns]
    
    # Drop rows with NaN values
    composite_df_scaled = composite_df_scaled.dropna()
    
    # Plot the correlation pairplot
    pairplot = sns.pairplot(composite_df_scaled, diag_kind='kde', plot_kws={'alpha': 0.5})
    
    return pairplot


def perform_pca_and_plot(df, components=['weight_avg_scaled', 'detour_ratio_scaled'], biplot=False, title="PCA Analysis"):
    """
    Performs PCA on the given data, generates a scree plot, and optionally creates a biplot.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the features for PCA.
        components (list): The list of column names to use for PCA.
        biplot (bool): If True, generates a PCA biplot (default is False).
        title (str): The title to display above the plots.
    
    Returns:
        tuple: PCA object and the PCA-transformed components (numpy array).
    """

    n_components = len(components)

    # Perform PCA
    data = df[components]

    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data)

    # Set up the figure layout
    if biplot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Side-by-side plots
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes = [axes]  # Ensure we always have a list of axes for uniform handling

    # Scree plot
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7, color='b')
    axes[0].set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    axes[0].set_xticklabels([f'PC{i}' for i in range(1, n_components + 1)])
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_title('Scree Plot')

    # Optional: Create a biplot
    if biplot:
        axes[1].scatter(pca_components[:, 0], pca_components[:, 1], alpha=0.5)
        
        # Plot arrows for each original variable
        for i, (x, y) in enumerate(zip(pca.components_[0], pca.components_[1])):
            axes[1].arrow(0, 0, x, y, color='r', alpha=0.8, head_width=0.05)
            axes[1].text(x * 1.15, y * 1.15, data.columns[i], color='g', fontsize=10)
        
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('PCA Biplot')
        axes[1].grid()

    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

    return pca, pca_components
