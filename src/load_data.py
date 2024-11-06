"""
Load the Data from the wikispeedia dataset.

Clay Foye, sep 2024
"""

import sys
import os
import urllib.parse
import pandas as pd

# Add the base project directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))



from Project.include.article import Article



def decode_df_unicode(df: pd.DataFrame) -> pd.DataFrame:
    """
    ChatGPT function to decode unicode in a dataframe.

    Arguments
    ---------
    df : pd.DataFrame
        DataFrame containing encoded unicode strings

    Returns
    -------
    pd.DataFrame
        df, but with the unicode strings parsed.
    """
    return df.map(lambda x: urllib.parse.unquote(x) if isinstance(x, str) else x)


def load_articles_into_df(
    articles_file_path: str = "Data/wikispeedia_paths-and-graph/articles.tsv",
) -> pd.DataFrame:
    """
    Load the "articles.tsv" file into a Pandas DF

    Arguments
    ---------
    articles_file_path : str
        The relative file path for the articles.tsv file

    Returns
    -------
    pd.DataFrame
        A dataframe containing articles.tsv with the unicode rendered.
    """
    articles = pd.read_csv(
        articles_file_path, sep="\t", comment="#", names=["article"], header=None
    )
    articles = decode_df_unicode(articles)

    return articles


def load_categories(
    categories_file_path: str = "Data/wikispeedia_paths-and-graph/categories.tsv",
) -> pd.DataFrame:
    """
    Load the Categories DataFrame
    """

    categories = pd.read_csv(
        categories_file_path,
        sep="\t",
        comment="#",
        names=["article", "category"],
        header=None,
    )

    categories = decode_df_unicode(categories)

    aggregated_categories = (
        categories.groupby("article")["category"].apply(list).reset_index()
    )

    return aggregated_categories


def load_links(
    links_file_path: str = "Data/wikispeedia_paths-and-graph/links.tsv",
) -> pd.DataFrame:
    """
    Load the Categories DataFrame
    """

    links = pd.read_csv(
        links_file_path,
        sep="\t",
        comment="#",
        names=["linkSource", "linkTarget"],
        header=None,
    )

    links = decode_df_unicode(links)

    aggregated_links = (
        links.groupby("linkSource")["linkTarget"].apply(list).reset_index()
    )

    return aggregated_links


def links_to_dict(links: pd.DataFrame) -> map:
    """
    Convert the links dictionary from 'load_links' to a python dictionary
    """
    return links.set_index("linkSource")["linkTarget"].to_dict()


def load_paths_finished(
    paths_finished_file_path: str = "Data/wikispeedia_paths-and-graph/paths_finished.tsv",
) -> pd.DataFrame:
    """
    Load the finished (successful) paths for the wikispeedia game
    """
    paths_finished = pd.read_csv(
        paths_finished_file_path,
        sep="\t",
        comment="#",
        names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"],
        header=None,
    )

    paths_finished = decode_df_unicode(paths_finished)

    return paths_finished


def load_paths_unfinished(
    paths_unfinished_file_path: str = "Data/wikispeedia_paths-and-graph/paths_unfinished.tsv",
) -> pd.DataFrame:
    """
    Load the unfinished (unsuccessful) paths for the wikispeedia game
    """

    paths_unfinished = pd.read_csv(
        paths_unfinished_file_path,
        sep="\t",
        comment="#",
        names=[
            "hashedIpAddress",
            "timestamp",
            "durationInSec",
            "path",
            "target",
            "type",
        ],
        header=None,
    )

    paths_unfinished = decode_df_unicode(paths_unfinished)

    return paths_unfinished


def load_path(path: str) -> list:
    """
    Convert a "path" string to a list of strings.
    This can be altered later to make a "path" object if we want.
    """

    return path.split(";")


def load_distances(
    file_path: str = "Data/wikispeedia_paths-and-graph/shortest-path-distance-matrix.txt",
) -> dict:
    """
    Load shortest-path distances into a dictionary of dictionaries,
    with faster parsing and reduced overhead.

    Each row corresponds to the shortest path distances from one source article
    to all target articles.

    Args:
        file_path (str): The path to the text file.

    Returns:
        distances (dict): A dictionary where keys are source articles and values
                          are dictionaries with target articles as keys and distances as values.
    """
    articles = (
        load_articles_into_df().article.tolist()
    )  # Convert articles to a list for faster access
    distances = {}
    article_idx = 0  # Track current source article index

    # Open the file and read line by line
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Skip comment lines or empty lines
            if line[0] == "#" or not line.strip():
                continue

            # Parse the line: replace underscores with None and convert digits to integers
            parsed_distances = {
                articles[idx]: None if char == "_" else int(char)
                for idx, char in enumerate(line.strip())
            }

            # Store distances for the current source article
            distances[articles[article_idx]] = parsed_distances
            article_idx += 1

    return distances


def load_article_objects(
    articles_file_path: str = "Data/wikispeedia_paths-and-graph/articles.tsv",
    raw_utf_keys=False,
) -> map:
    """
    Load all of the articles into "Article" objects, and return a dict of all the article objects.

    Arguments
    ---------
    articles_file_path : str
        The file path to the "articles.tsv" file
    raw_utf_keys : bool
        Defaults False. If true, store the articles in the dictionary with unrendered UTF. Otherwise, rendered UTF.

    Returns
    -------
    article_objects : map
        A dictionary with:
            keys : str
                article names in either rendered or unrendered UTF
            values : Article
                The corresponding article object
    """

    # Load in articles, distances, links, and categories.
    articles = pd.read_csv(
        articles_file_path, sep="\t", comment="#", names=["article"], header=None
    )
    distances = load_distances()
    categories = load_categories()
    links = load_links()

    ### Warning for print statement of articles without categories
    print("---- WARNING: The following articles have a problem: ----")

    article_objects = {}

    for i, article in articles.iterrows():
        # Render article name in utf
        article_utf_name = urllib.parse.unquote(article.article)

        # Load in categories and links
        article_categories = categories.loc[
            categories.article == article_utf_name
        ].category.values
        article_links = links.loc[
            links.linkSource == article_utf_name
        ].linkTarget.values

        # Warning messages about articles
        if len(article_categories) == 0:
            print(f"{article.article} has no category listed.")

        if len(article_links) == 0:
            print(f"{article.article} has no links (out-edges) listed.")

        # Check whether to use raw UTF or Rendered UTF for keys
        article_key = article.article if raw_utf_keys else article_utf_name

        # Instantiate and Store the Article object
        article_objects[article_key] = Article(
            name=article_utf_name,
            text_filepath=f"Data/plaintext_articles/{article.article}.txt",
            distances=distances[article_utf_name],
            links=article_links[0] if len(article_links) > 0 else None,
            category=article_categories[0] if len(article_categories) > 0 else None,
        )

    # Return the dict of article_objects
    return article_objects