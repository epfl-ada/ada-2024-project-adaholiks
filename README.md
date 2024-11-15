# The good, the bad and the ugly: the impact of articles on wikispeedia path success

## Abstract

In an age of rapid information consumption, understanding how specific article characteristics influence navigation behaviour in Wikispeedia could offer insights into decision-making within hyperlinked environments. We hypothesise that player success in reaching target articles in Wikispeedia may be impacted by inherent article attributes, such as length, link density, and topic specificity. Our project seeks to analyse how these attributes correlate with path success. First, metrics are established for quantifying article “goodness”. This is achieved both using a theoretical and player-centred approach, resulting in article rankings. Subsequently, we aim to investigate what article characteristics influence player success, revealing patterns that could enrich our understanding of strategic decision-making within web-based information pathways. Uncovering these patterns could offer broader implications for user engagement and navigation in digital environments.

## Research Questions

This project aims to address the following research questions:

- How can one determine what constitutes a "good" article in Wikispeedia?
- Are the articles that players flock to in the game "good" in the eyes of graph theory?
- Do players exhibit biases in terms of the articles they tend to click on and if so, what effect does this have on success?
- What role does semantic distance play in navigation success?

## Methodology

The following methodology would be used to conduct this project work. No additional datasets will be used in this project. %to be confirmed

### Part 0 - Initial Data Preparation

The dataset is organized into two primary data frames:

- A data frame containing information about individual articles and their descriptive attributes.
- A data frame detailing the various paths, including computed shortest paths and player paths, categorized into successful and failed attempts.

### Part 1 - Data Analysis and Exploration

The two constructed data frames are analysed using statistical methods to uncover key patterns and characteristics. This analysis focuses on:

- How can one determine what constitutes a "good" article in Wikispeedia?

### Part 2 - Graph theory based top articles

In this section, we focus on the theoretically best articles in the network by using graph theory and computation of shortest paths. Here, we define as a good article, one that appears more often than not in the shortest paths between any given source-target pair.

### Part 3 - Game based top articles

### - Description of article attributes

Articles are analysed for different attributes both theoretically and in the eyes of the player. The theoretical analysis is performed using the shortest path matrix to determine how present articles are in the shortest possible paths. Metrics from graph theory such as between centrality, cut vertices and page rank are extracted from this analysis. An additional semantic analysis of the different articles is also performed. The player centred analysis on the other hand is performed on successful and unsuccessful player paths and seeks to measure the prevalence of certain articles depending on player centred attributes such as hyperlink density, article quality and thematic centrality. Two rankings of articles are obtained and analysed in order to reveal how their attributes impact different paths and to reveal underlying player biases.

### - Comparison of Article Rankings

The final article rankings obtained at the end of part 1 are subsequently analysed in order to obtain the differences between the theoretical and player centred good articles. The results of this analysis will reveal player biases for certain types of articles.

\newline We aim to get to the end of part 2 for project milestone 2.

### - Tf we doing after this actually, I still haven't really clocked

### - Result Presentation and Data Story

The final step of our project will involve clearing up our work and git repository as well as create a data story and its associated webpage.

## Team Organisation and Timeline

As previously stated, parts 0 through 2 are performed for project milestone 2. The following parts will be performed according to the timeline below:

- Part 3  ....
- ...
- Part ? ...

Our team organisation is as follows:

- Clay:
- Finn:
- Fletcher:
- Karl:
- Oscar:

- `name` (str): The name of the article.
- `text_filepath` (str): File path to the article's plain text.
- `text` (str): The actual text of the article, which is read from the file.
- `distances` (dict): A dictionary mapping other articles to their shortest-path distances from this article.
- `links` (list): Outbound links from the article (connections to other articles).
- `category` (str): The category of the article.
