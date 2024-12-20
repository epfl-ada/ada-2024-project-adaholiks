# The Good, the Bad, and the Ugly: The Impact of Articles on Wikispeedia Path Success

## Abstract

In an age of rapid information consumption, understanding how specific article characteristics influence navigation behavior in Wikispeedia could offer insights into decision-making within hyperlinked environments. We hypothesize that player success in reaching target articles in Wikispeedia may be impacted by inherent article attributes, such as length, link density, and topic specificity. Our project seeks to analyze how these attributes correlate with path success. First, metrics are established for quantifying article “goodness”. This is achieved both using a theoretical and player-centered approach, resulting in article rankings. Subsequently, we aim to investigate what article characteristics influence player success, through the establishment of statistically significant positive correlation to good articles. These characteristics are then converted into attributes that serve as inputs for supervised machine learning models, where the target variable is goodness, revealing patterns that could enrich our understanding of strategic decision-making within web-based information pathways.

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
  We first look at articles that come up the most in the played path data after outlier removal. This serves as comparison for the other approaches.
  
- **Weighted approach by difference to optimal length**  
  The idea is to assign each played path a weight based on the difference between its length and the shortest length from the start to the target article. An article goodness score is then be computed using both a weighted average, or a sum of centred averages.
  

**Next steps**

Thus far, we have only really considered one aspect of the paths in order to define good articles. That is the difference in the path length taken and shortest distance. However, there are a few other aspects that we can potentially also consider. For example, when articles are overrepresented in unfinished paths or associated with a lot of back clicks, the ‘goodness’ score should be lower. A more sophisticated approach could thus look like this:
- **Scoring function**: `alpha*dist_diff + beta*unfinished_penalty + gamma*back_click_unfinished`. The parameters can be tuned through machine learning (e.g regression).
- We have also noticed that certain start target pairs have over a thousand played paths. This could be used to conduct much more robust scoring of the articles in question and potentially to train the aforementioned weights.

### Part 4 - Analysis of the Results Found and Correlation with Attributes
Extracted attributes from part 2 such as incoming links, outgoing links, article length, and hyperlink density can be taken into account and correlated with our article rankings from part 3. Other attributes such as category centrality, position of hyperlinks, and semantic distance can also be extracted and correlated with path success metrics.


### Part 4a - Bot Generated Paths and Correlation with Attributes
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

### Part 5 - Machine Learning Approach
It is then possible to do a more sophisticated quantification of the different attributes using machine learning. Labels are the computed article scores and are predicted using the article attributes. We will start by implementing a simple linear regression model. Depending on the findings, we will potentially consider more complex supervised models.

## Team Organization and Timeline
- **Clay**: Step 4
- **Finn**: Complete Step 3
- **Fletcher**: Complete Step 3 & 5
- **Karl**: Step 5
- **Oscar**: Complete Step 4

Our general timeline is to have steps 1 through 3 completed by the end of week 10 while concurrently doing the homework. Parts 4 and 5 would then be completed over weeks 11 and 12, leaving us a final week to do the data story.


