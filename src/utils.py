"""
Load the Data from the wikispeedia dataset.

Fletcher Collis, nov 2024
"""

import sys
import os
import pandas as pd

# Add the base project directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

########################################################################################################################
# Functions written for "Handling the data by loading the Dataset into 2 Dataframes"


# for each row in article_dataframe, 
# load the plain_text from the file `Data/plaintext_articles/{article_name}.txt`
def load_plain_text(article_name):
    file_path = f'Data/plaintext_articles/{article_name}.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

########################################################################################################################
# Functions written for "Statistical analysis of the player paths data"


# Here describe function

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

########################################################################################################################
# Functions written for "Statistical analysis of the article data"


########################################################################################################################
# Functions written for "Part 1 : graph theory based top articles."


########################################################################################################################
# Functions written for "Part 2 : game based top articles"


########################################################################################################################
# Functions written for "Part 3 : Looking at the attributes that make an article "good" for the game"