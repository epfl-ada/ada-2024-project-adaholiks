# The Good, the Bad, and the Ugly: The Impact of Articles on Wikispeedia Path Success

## Abstract

In an age of rapid information consumption, understanding how specific article characteristics influence navigation behavior in Wikispeedia could offer insights into decision-making within hyperlinked environments. We hypothesize that player success in reaching target articles in Wikispeedia may be impacted by inherent article attributes, such as length, link density, and topic specificity. Our project seeks to analyze how these attributes correlate with path success. First, metrics are established for quantifying article “goodness.” This is achieved using both theoretical and player-centered approaches, resulting in article rankings. We aim to investigate which article characteristics influence player success, revealing patterns that could enrich our understanding of strategic decision-making within web-based information pathways. Uncovering these patterns could offer broader implications for user engagement and navigation in digital environments.

## Research Questions

This project aims to address the following research questions:

1. How can one determine what constitutes a "good" article in Wikispeedia?
2. Are the articles that players tend to flock to in the game "good" from the perspective of graph theory?
3. What attributes make an article a driver for efficient path completion in Wikispeedia?

## Methodology

### Part 0 - Initial Data Preparation

The dataset is organized into two primary data frames:
- A data frame containing information about individual articles and their descriptive attributes.
- A data frame detailing various paths, including computed shortest paths and player paths, categorized into successful and failed attempts.

### Part 1 - Data Analysis and Exploration

The two constructed data frames are analyzed using statistical methods to uncover key patterns and characteristics. This analysis focuses on:
- Attributes of individual articles.
- Traits of played paths (successful and failed) to gain deeper insights into the dataset.

In addition, we explore the semantic distance between articles. One approach to compute semantic distance is by creating embeddings for the article titles. Initially, we compute basic semantic distance by generating embeddings for the titles. In the future, we plan to expand this by calculating embeddings based on the full text of each article and using them as another type of attribute.

Dimensionality reduction techniques such as PCA and T-SNE will be utilized to extract meaningful representations from high-dimensional data. For preliminary analysis, we can use TensorFlow’s Embedding Projector tool for visualization and analysis.

### Part 2 - Graph Theory-Based Top Articles

In this phase, we identify the theoretically "best" articles in the network using graph theory principles. We define a "good" article as one that frequently appears in the shortest paths between any given source-target pair played in the game. To achieve this:
- We calculate the frequency of each article's appearance across all shortest paths for each possible source-target pair played in the game.
- We normalize this count by dividing it by the total number of shortest paths for each source-target pair to ensure that the importance reflects the article's overall significance in the network.

### Part 3 - Game-Based Top Articles

We analyze the paths taken by players to determine the most valued articles in actual gameplay using different approaches:
1. **Naive approach**: Identify articles that appear most frequently in filtered data. This serves as a baseline comparison.
2. **Weighted average of articles**: Explanation...
3. **Weighted sum of sampled articles**: Explanation...

Further considerations for path characteristics, such as unfinished paths or back clicks, will be used to refine article scores.

### Part 4 - Analysis of Results and Correlation with Attributes

We will correlate attributes extracted from Parts 2 and 3 (e.g., incoming/outgoing links, article length, hyperlink density) with article rankings. Additional attributes such as category centrality, hyperlink position, and semantic distance will also be examined for correlation with path success metrics.

### Part 5 - Machine Learning Approach

We will use machine learning to quantify the relationship between article attributes and computed article scores. We will start with a simple linear regression model and potentially explore more complex supervised models.

##

