# The Good, the Bad, and the Ugly: The Impact of Articles on Wikispeedia Path Success

## Abstract

In an age of rapid information consumption, understanding how specific article characteristics influence navigation behavior in Wikispeedia can provide valuable insights into decision-making within hyperlinked environments. We hypothesize that a player's success in reaching target articles in Wikispeedia may be influenced by inherent article attributes, such as length, link density, and topic specificity. Our project aims to analyze how these attributes correlate with path success. First, we establish scores to quantify articles based on how effectively they help players navigate the network. This is done using both a theoretical approach and a player-centered perspective, resulting in article rankings. Next, we investigate which article characteristics contribute to player success by establishing statistically significant positive correlations with high-performing articles. These characteristics are then transformed into features that serve as inputs for supervised machine learning models, with the article score as the target variable. The goal is to uncover patterns that can enhance our understanding of strategic decision-making in web-based information pathways.

## Research Questions
This project aims to address the following research questions:
- How can one determine what constitutes a "good" article in Wikispeedia?
- Are the articles that players tend to flock to in the game "good" from the perspective of graph theory?
- What attributes make an article a driver for efficient path completion in Wikispeedia?

## Methodology

### Part 0 - Initial Data Preparation
The dataset is organized into two primary data frames:
- A data frame containing information about individual articles and their descriptive attributes.
- A data frame detailing the various paths, including computed shortest paths and player paths, categorized into successful and failed attempts.

### Part 1 - Data Analysis and Exploration
The two constructed data frames are analyzed using statistical methods to uncover key patterns and characteristics. This analysis focuses on:
- Attributes of individual articles.
- Traits of played paths (successful and failed) to gain deeper insights into the dataset.

In addition to these analyses, we explored the semantic distance between articles. One approach to compute semantic distance is by creating embeddings for the article titles. We computed basic semantic distance by generating embeddings for the titles.There are several tools available to visualize and analyze these embeddings. A common strategy to extract meaningful representations from high-dimensional data is to reduce the dimensionality of the embeddings. Two popular techniques for dimensionality reduction are principal component analysis (PCA) and T-SNE. (For a deeper explanation of T-SNE, see this overview.) We can use both PCA and T-SNE through libraries like scikit-learn.

Using T-SNE via TensorFlow's Embedding Projector, we visually confirm that the title embeddings are an effective instrumentalization of categories. We originally planned to use the embeddings of the entire article text, but this was discarded due to the unnecessary amount of noise for diminishing returns. These full-text embeddings would have also been too unweidly.

These embeddings proved to be quite useful. We derived multiple different attributes from the cosine distance of a current article to the articles it links to. We also created an algorithm to play a Wikispeedia path by greedily traversing the Wikispeedia graph due to cosine distance of the next articles and the target article. Our attributes correlated better with the scores extracted from these paths generated by greedy embedding traversals than the scores extracted from human paths. This led us to the conclusion that article quality is player-specific, and cannot be sufficiently generalized from in-article attributes.

### Part 2 - Game-Based Top Articles

This section analyzes player paths to identify the most valued articles during gameplay, using approaches ranging from simple to advanced:

#### **1. Weighted Path Length Ratio**
We calculate article scores by weighting paths based on the ratio of the shortest possible path length to the actual path length. Scores are computed as:
- **Weighted Average:** Average of path weights per article.
- **Sum of Centered Averages:** Total deviation from mean weights for paths containing the article.

#### **2. Complementary Scores**
To refine the analysis, we add:
- **Detour Ratio:** Penalizes articles often visited during detours.
- **Unfinished Ratio:** Penalizes articles overrepresented in incomplete paths.

#### **3. Composite Scoring Function**
We combine the above metrics using:

composite score function : w1*path_length_ratio - w2*unfinished_ratio - w3*detour_ratio.

Weights w1, w2, and w3 are defined heuristically.

#### **4. Speed-Based Scores**
Finally, we reward articles frequently appearing in paths completed quickly to account for the speed aspect of "Wikispeedia."

This scoring framework provides a nuanced understanding of article importance in gameplay.
  
### Part 3 - Graph Theory-Based Top Articles
In this phase, we identify the theoretically "best" articles in the network using principles in graph theory, this allows to get a comparison to the computed scores from the previous part. We define a "good" article as one that frequently appears in the shortest paths between any given source-target pair. To do this, we calculate the frequency of each article's appearance across all shortest paths for each possible source-target pair played in the game. By doing this we obtain a list of theoretically good articles that are good for navigating through the network. We have also divided this count by the total number of shortest paths for each source-target pair. This normalization helps us measure the true importance of each article in the network. Without normalization, an article that appears in every shortest path for a specific source-target pair might seem highly important, even if it doesn’t play a critical role across the entire network. By normalizing, we avoid overemphasizing articles that are only crucial within specific source-target pairs and ensure that the count more accurately reflects an article’s overall significance in the network.

### Part 4 - Attribute-Scoe Correlation Analysis

We analyze the relationship between extracted attributes (e.g., incoming links, outgoing links, article length, and hyperlink density) and the computed scores from Part 2, focusing separately on path length and path speed. Additional attributes, such as category centrality, hyperlink positions, and semantic distance, are also examined for their potential correlations with path scores. This comprehensive analysis aims to uncover key factors influencing article performance in navigation tasks.


### Part 5 - Bot Generated Paths and Correlation with Attributes
After being unsatisfied with the low correlation between our attributes and extracted scores, we decided to create new paths according to a single Wikispeedia strategy.

Let's use the embeddings from our data analysis to help us define a greedy walk of the wikispeedia graph.

```
    For each start, end in paths:
    current_article = start
    while current_article != end:
        Out of the possible next articles, choose the article that satisfies the following conditions:
            1. we have not visited
            2. has the smallest cosine distance to the end of the path
        If no articles satisfy this, discard the path.
```

In the end, it turned out that the sum of centered weights and average weight scores computed from these "greedy" paths were useful. Our attributes correlated more closely with these paths than with the generalized player paths. From this we were able to extract the understanding that there *is no such thing as a good article* in a general sense. What makes an article *good* or *bad* depends on a player's strategy. The generalized human-created paths have paths created by many different archetypes of players, so it is no wonder our attributes were not able to correlate closely with the scores extracted from those paths.

### Part 6 - Machine Learning Approach
In P2 we discussed possible Machine Learning approaches. These were attempted, but showed no significant results. We detail one of those implementations at the end of the P3 Notebook. The lack of significant results is unsurprising given the low correlation values of our attributes with scores extracted from human path data.

## Team Organization
- **Clay**: Semantic Analysis, Algorithm Creation, Optimization, Documentation
- **Finn**: Path Analysis, Statistical Analysis, Algorithm Creation, Documenation
- **Fletcher**: Website Creation, Graphics and Plot Creation, Documentation, Graph Algorithms
- **Karl**:  Website Creation, Graphics and Plot Creation, Documentation, Problem Formulation
- **Oscar**: Statistical Analysis, Machine Learning and Regression, Correlation Analysis, Documentation


