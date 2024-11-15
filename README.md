# The Good, the Bad, and the Ugly: The Impact of Articles on Wikispeedia Path Success

## Abstract
In an age of rapid information consumption, understanding how specific article characteristics influence navigation behavior in Wikispeedia could offer insights into decision-making within hyperlinked environments. We hypothesize that player success in reaching target articles in Wikispeedia may be impacted by inherent article attributes, such as length, link density, and topic specificity. Our project seeks to analyze how these attributes correlate with path success. First, metrics are established for quantifying article “goodness”. This is achieved both using a theoretical and player-centered approach, resulting in article rankings. Subsequently, we aim to investigate what article characteristics influence player success, revealing patterns that could enrich our understanding of strategic decision-making within web-based information pathways. Uncovering these patterns could offer broader implications for user engagement and navigation in digital environments.

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

In addition to these analyses, we explore the semantic distance between articles. One approach to compute semantic distance is by creating embeddings for the article titles. Initially, we can compute basic semantic distance by generating embeddings for the titles. In the future, we plan to expand this by calculating embeddings based on the full text of each article and use it as another type of attribute. There are several tools available to visualize and analyze these embeddings. A common strategy to extract meaningful representations from high-dimensional data is to reduce the dimensionality of the embeddings. Two popular techniques for dimensionality reduction are principal component analysis (PCA) and T-SNE. (For a deeper explanation of T-SNE, see this overview.) We can use both PCA and T-SNE through libraries like scikit-learn. However, for this stage, our goal is to get a preliminary understanding of the data. To achieve this, we can use TensorFlow’s Embedding Projector tool. This tool can be run locally or used directly in a browser, providing a visual interface to explore and analyze the article embeddings.

### Part 2 - Graph Theory-Based Top Articles
In this phase, we identify the theoretically "best" articles in the network using principles in graph theory. We define a "good" article as one that frequently appears in the shortest paths between any given source-target pair. To do this, we calculate the frequency of each article's appearance across all shortest paths for each possible source-target pair played in the game. By doing this we obtain a list of theoretically good articles that are good for navigating through the network. We have also divided this count by the total number of shortest paths for each source-target pair. This normalization helps us measure the true importance of each article in the network. Without normalization, an article that appears in every shortest path for a specific source-target pair might seem highly important, even if it doesn’t play a critical role across the entire network. By normalizing, we avoid overemphasizing articles that are only crucial within specific source-target pairs and ensure that the count more accurately reflects an article’s overall significance in the network.

### Part 3 - Game-Based Top Articles
In this section, we concentrate on the player data and analyze the paths taken to determine the most valued articles by the players in actual gameplay. We quantified the level of importance by using various approaches, from simple to more complex:
- **Naive approach**  
  We first look at articles that come up the most in filtered data (explain here what filtered data is). This is of course a naive approach and will serve as a comparison for other metrics.
  
- **Weighted approach by difference to optimal length**  
  The idea is to assign each played path a weight based on the difference between its length and the shortest length from the start to the target article. An article goodness score can then be computed using either a weighted average:
  
  \[
  \text{avg\_weight} = \frac{\sum \text{weight}}{n\_appearances}
  \]
  
  or a centered weighted sum:
  
  \[
  \sum (\text{weight} - \text{avg\_weight})
  \]

Next steps: Thus far, we have only really considered one aspect of the paths in order to define good articles. That is the difference in the path length taken and shortest distance. However, there are a few other aspects that we can potentially also consider. For example, when articles are overrepresented in unfinished paths or associated with a lot of back clicks, the ‘goodness’ score should be lower. A more sophisticated approach could thus look like this:
- **Scoring function**: `alpha*dist_diff + beta*unfinished_penalty + gamma*back_click_unfinished`. The parameters can be determined through machine learning (e.g regression).
- We noticed that a few start-target pairs have over a thousand played paths. On a subset of articles that show up in these paths, we could have much more robust scoring… Can also be used to train the weights of the above…

### Part 4 - Analysis of the Results Found and Correlation with Attributes
Extracted attributes from part 2 such as incoming links, outgoing links, article length, and hyperlink density can be taken into account and correlated with our article rankings from part 3. Other attributes such as category centrality, position of hyperlinks, and semantic distance can also be extracted and correlated with path success metrics.

### Part 5 - Machine Learning Approach
It is then possible to do a more sophisticated quantification of the different attributes using machine learning. Labels are the computed article scores and are predicted using the article attributes. We will start by implementing a simple linear regression model. Depending on the findings, we will potentially consider more complex supervised models.

## Team Organization and Timeline
- **Clay**: Step 4
- **Finn**: Complete Step 3
- **Fletcher**: Complete Step 3 & 5
- **Karl**: Step 5
- **Oscar**: Complete Step 4

Our general timeline is to have steps 1 through 3 completed by the end of week 10 while concurrently doing the homework. Parts 4 and 5 would then be completed over weeks 11 and 12, leaving us a final week to do the data story.


