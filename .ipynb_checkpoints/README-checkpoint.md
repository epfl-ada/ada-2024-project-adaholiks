# Data Download and Loading

This is a small project to make loading the Wikispeedia Data much easier. Included are several utility functions, as well as an implementation of an Article class. Eventually, we should write some code to generate NetworkX Nodes/ out of Article objects.

I also included an unfinished 'Path' class that will be useful later if we want to do specific operations on paths.

All of this is the first step in our data pipeline. The next step will be to connect it to some analysis tools. The main graphical analysis tool for python is [NetworkX](https://networkx.org/documentation/stable/reference/generators.html). The Article objects also load their texts, so we can do some textual analysis on them as well.


### Directory Structure

The directory structure for the `Project` folder should be:

```
.
├── Data
│   ├── plaintext_articles
│   ├── wikispeedia_articles_plaintext.tar.gz
│   ├── wikispeedia_paths-and-graph
│   └── wikispeedia_paths-and-graph.tar.gz
├── README.md
├── include
│   ├── __pycache__
│   ├── article.py
│   └── path.py
└── src
    ├── __pycache__
    └── load_data.py
```
So download and unpack the data accordingly.

## Data Loading Utilities

All of the data loading utilities are in `Project/src/load_data.py`, The functions are:

- `decode_df_unicode(df: pd.DataFrame) -> pd.DataFrame`
    - This function decodes any Unicode-encoded strings in a Pandas DataFrame.
- `load_articles_into_df(articles_file_path: str) -> pd.DataFrame`
    - This function loads articles from the articles.tsv file into a Pandas DataFrame, decoding any Unicode strings in the process.
- `load_categories(categories_file_path: str) -> pd.DataFrame`
    - Loads and decodes categories associated with articles, grouping them by article.
- `load_links(links_file_path: str) -> pd.DataFrame`
    - Loads and decodes links between articles from the links.tsv file.
- `links_to_dict(links: pd.DataFrame) -> dict`
    - Converts a DataFrame of links into a dictionary.
- `load_paths_finished(paths_finished_file_path: str) -> pd.DataFrame`
    - Loads the finished (successful) paths from the paths_finished.tsv file.
- `load_paths_unfinished(paths_unfinished_file_path: str) -> pd.DataFrame`
    - Loads the unfinished (unsuccessful) paths from the paths_unfinished.tsv file.
- `load_path(path: str) -> list`
    - Converts a path string into a list of article names.
- `load_distances(file_path: str) -> dict`
    - Loads shortest-path distances between articles from a distance matrix file into a dictionary of dictionaries.
- `load_article_objects(articles_file_path: str, raw_utf_keys: bool = False) -> dict`
    - Loads articles into Article objects, with optional control over whether to use rendered or unrendered Unicode.

All these are useful in their own way! I think `load_article_objects` is the most extensive due to its use of the Article object.

## Article Class


### Using the Article Class
The Article class is designed to represent a Wikipedia article as an object in the Wikispeedia dataset. Each Article instance holds key properties like its name, category, links to other articles, distances to other articles, and its text content.

### Properties:
- `name` (str): The name of the article.
- `text_filepath` (str): File path to the article's plain text.
- `text` (str): The actual text of the article, which is read from the file.
- `distances` (dict): A dictionary mapping other articles to their shortest-path distances from this article.
- `links` (list): Outbound links from the article (connections to other articles).
- `category` (str): The category of the article.