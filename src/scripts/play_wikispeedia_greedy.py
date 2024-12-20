"""
Play Wikispeedia Greedily:

For the existing filtered paths, play by choosing the next article with the smallest distance to the target article.

"""
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from src.utils.filtering_utils import IQR_filtering




def create_distance_matrix(articles_df : pd.DataFrame) -> tuple[dict, dict, dict, list]:
    
    # Store the graph, embeddings in dictionary
    next_article_dict  = {}
    embedding_dict  = {}

    for index, row in articles_df.iterrows():
        next_article_dict[row['article']] = row['linkTarget']
        embedding_dict[row['article']] = row['embeddings']


        
    all_article_names = list(set(articles_df['article'].to_list()))
    num_unique_articles = len(all_article_names)


    index_lookup = {string: index for index, string in enumerate(all_article_names)}

    index_lookup['Long_peper'] = index_lookup['Long_pepper']
    embedding_dict['Long_peper'] = embedding_dict["Long_pepper"]
    # Create a matrix to store the embeddings
    embeddings_matrix = np.zeros((num_unique_articles, len(embedding_dict[all_article_names[0]])))

    # Populate the embeddings matrix with the correct embeddings
    for article, index in index_lookup.items():
        embeddings_matrix[index] = embedding_dict[article]

    # Compute the cosine distances between all embeddings
    distance_matrix = cosine_distances(embeddings_matrix)

    return index_lookup, distance_matrix, next_article_dict, all_article_names



# Function to look up the distance using the indices from index_lookup
def get_distance(article1, article2, index_lookup, distance_matrix):
    index1 = index_lookup[article1]
    index2 = index_lookup[article2]
    return distance_matrix[index1, index2]



###  Play a list of paths with embedding lists. Get the DF
def play_path_list(articles_df : pd.DataFrame, path_list: list[tuple]):
    # Placeholder for storing all paths
    paths_data = []
    dead_ends = 0

    dead_end_articles = set()
    index_lookup, distance_matrix, next_article_dict, all_article_names = create_distance_matrix(articles_df)


    # Iterate over all paths with a progress bar
    for path_start, path_end in tqdm(path_list, desc="Processing paths"):
        if not path_end in all_article_names:
            continue
        current_article = path_start
        path = [current_article]
        visited_articles = set(path)  # Keep track of visited articles

        while current_article != path_end:
            # Get the list of next articles excluding already visited ones
            next_articles = [
                article
                for article in next_article_dict[current_article]
                if article not in visited_articles
            ]

            if not next_articles:
                # If there are no unvisited next articles, break to avoid infinite loop
                # print(f"Dead-end reached from {current_article}.")
                dead_end_articles.add(current_article)
                dead_ends += 1
                break

            # Compute distances to the target for all valid next articles
            distances = [get_distance(article, path_end, index_lookup, distance_matrix) for article in next_articles]
            # Find the next article with the minimum distance
            current_article = next_articles[np.argmin(distances)]
            path.append(current_article)
            visited_articles.add(current_article)  # Add the article to the visited set

        # Append path data to the list
        paths_data.append(
            {
                "start_article": path_start,
                "target_article": path_end,
                "path": path,
                "distance": articles_df[articles_df["article"] == path_start][
                    "distances"
                ].values[0][path_end],
            }
        )

    # Convert collected data to a DataFrame
    greedy_embedding_paths = pd.DataFrame(paths_data)

    print("Paths processed and stored successfully!")
    print(f"#{dead_ends} of Dead Ends")

    return greedy_embedding_paths


def filter_greedy_paths(greedy_embedding_paths : pd.DataFrame):
    
    # Set the random seed for reproducibility
    np.random.seed(42)
    threshold = 10
    # Shuffle the DataFrame
    shuffled_df = greedy_embedding_paths.sample(frac=1).reset_index(drop=True)

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

    IQR_filtered_greedy_paths, _ = IQR_filtering(downsampled_df, column='path_lengths')

    return IQR_filtered_greedy_paths

def average_outgoing_cosine_distance(
    articles_df: pd.DataFrame,
    n: int = 5
) -> pd.DataFrame:
    """
    Adds the average of the top n maximum and top n minimum cosine distances
    from each article's embedding to its outgoing linked articles.

    Parameters:
    - articles_df (pd.DataFrame): DataFrame containing articles with an 'article' column.
    - index_lookup (dict): Dictionary mapping article names to their indices in the distance_matrix.
    - distance_matrix (np.ndarray): Precomputed cosine distance matrix.
    - next_article_dict (dict): Adjacency dictionary mapping articles to their outgoing linked articles.
    - n (int): Number of top maximum and minimum distances to average.

    Returns:
    - pd.DataFrame: Updated articles_df with 'average_max_cosine_distance' and 'average_min_cosine_distance' columns.
    """
    index_lookup, distance_matrix, next_article_dict, all_article_names = create_distance_matrix(articles_df)


    # Initialize lists to store the average distances
    avg_max_distances = []
    avg_min_distances = []

    # Iterate over each article with a progress bar for large datasets
    for article in tqdm(articles_df['article'], desc="Calculating average distances"):
        # Initialize list to hold distances for the current article
        cosine_distances = []

        # Retrieve outgoing articles; handle cases where the article might have no outgoing links
        outgoing_articles = next_article_dict.get(article, [])

        for outgoing_article in outgoing_articles:
            # Retrieve indices; handle cases where the outgoing article might not be in index_lookup
            idx_current = index_lookup.get(article, None)
            idx_outgoing = index_lookup.get(outgoing_article, None)

            if idx_current is None or idx_outgoing is None:
                # Skip if either article is not found in the index_lookup
                continue

            # Retrieve the precomputed cosine distance
            distance = distance_matrix[idx_current, idx_outgoing]
            cosine_distances.append(distance)

        if cosine_distances:
            # Convert to NumPy array for efficient computation
            distances_array = np.array(cosine_distances)

            # Calculate the top n maximum distances
            if len(distances_array) >= n:
                max_distances = np.sort(distances_array)[-n:]
            else:
                max_distances = distances_array  # If fewer than n distances, take all

            # Calculate the top n minimum distances
            if len(distances_array) >= n:
                min_distances = np.sort(distances_array)[:n]
            else:
                min_distances = distances_array  # If fewer than n distances, take all

            # Compute the averages
            avg_max = np.mean(max_distances)
            avg_min = np.mean(min_distances)

            avg_max_distances.append(avg_max)
            avg_min_distances.append(avg_min)
        else:
            # Assign NaN if no distances are available
            avg_max_distances.append(np.nan)
            avg_min_distances.append(np.nan)

    # Assign the computed averages to the DataFrame
    articles_df['average_max_cosine_distance'] = avg_max_distances
    articles_df['average_min_cosine_distance'] = avg_min_distances

    return articles_df
