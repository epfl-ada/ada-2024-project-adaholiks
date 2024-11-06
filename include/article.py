"""
Article Class file

Clay Foye, sep 2024
"""

class Article:
    """
    The Article class is meant to represent a single article.
    Making articles into objects/classes is a nice feature if we want to write methods on them.

    There are specific articles in the Wikispeedia dataset which don't contain either:
        - category
        - links
        - text
        - etc
    In any case where an attribute/property is not found in the dataset, that attribute is set to None on the Attribute object.

    Properties
    -----------
    name : str
        The name of the article
    text_filepath : str
        The fp of the text of the article
    text : str
        the plaintext of the article
    distances : dict
        the distances to all other articles from this article
    links : list
        All the links of this article (out-edges)
    category : str
        the category of the article

    """


    def __init__(self, name : str, text_filepath : str, distances : map, links: list, category : list):
        self.name = name
        self.text_filepath = text_filepath
        self.distances = distances
        self.links = links
        self.category = category

    @property
    def text(self):
        try:
            with open(self.text_filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError as e:
            print(f"Error: The file '{self.text_filepath}' was not found. Please check the file path and try again.")
            return None


