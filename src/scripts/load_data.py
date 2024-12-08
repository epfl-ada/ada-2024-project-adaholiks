from src.utils.load_utils import *


def load_data():
    """
    Ensures necessary directories exist, loads data into dataframes, and prints their shapes.

    Returns:
        tuple: A tuple containing the loaded dataframes in the following order:
               (articles, categories, links, paths_finished, paths_unfinished, distances)
    """
    # Ensure the directory exists
    data_dir = 'Data/dataframes'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Directory '{data_dir}' created successfully.")
    else:
        print(f"Directory '{data_dir}' already exists.")

    # Load the data
    articles = load_articles_into_df()
    categories = load_categories()
    links = load_links()
    paths_finished = load_paths_finished()
    paths_unfinished = load_paths_unfinished()
    distances = load_distances()

    # Print the shapes of the loaded data
    print("Articles shape: ", articles.shape)
    print("Categories shape: ", categories.shape)
    print("Links shape: ", links.shape)
    print("Paths finished shape: ", paths_finished.shape)
    print("Paths unfinished shape: ", paths_unfinished.shape)

    # Return the loaded dataframes
    return articles, categories, links, paths_finished, paths_unfinished, distances
