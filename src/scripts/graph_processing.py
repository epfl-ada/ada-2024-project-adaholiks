import networkx as nx
import pandas as pd
from collections import defaultdict

def compute_shortest_path_counts(articles, pair_counts):
    """
     Constructs a directed graph from the provided articles and computes the normalized shortest path counts.

     For each source-target pair in pair_counts, all shortest paths are extracted from the graph.
     Each intermediate article in these paths contributes a fraction equal to 1/(number_of_shortest_paths_for_that_pair)
     to its shortest path count. By summing these fractions over all pairs, we get a measure of how important
     each article is across the entire network of shortest paths.

     Parameters
     ----------
     articles : dict
         A dictionary where keys are article names and values are article objects. Each article object should
         have a 'category' attribute (string) and a 'links' attribute (list of outgoing link names).
     pair_counts : pd.DataFrame
         A DataFrame with 'start_article' and 'target_article' columns, representing the source-target pairs
         for which shortest paths are considered.

     Returns
     -------
     node_counts_df : pd.DataFrame
         A DataFrame with ['Node', 'ShortestPathCount'] columns, sorted by 'ShortestPathCount' in descending order.
         'ShortestPathCount' reflects the normalized importance of each article across all shortest paths
         between the given source-target pairs.
     errors : int
         The number of source-target pairs for which no shortest path exists.
     all_nb_of_shortest_paths : list
         A list containing the count of shortest paths found for each source-target pair processed.
     """
    G = nx.DiGraph()

    for article_name, article_obj in articles.items():
        # Add the article as a node with its category
        G.add_node(article_name, category=article_obj.category)

        # Add directed edges for outgoing links
        if article_obj.links:
            for link in article_obj.links:
                if link in articles:  # Ensure linked article exists
                    G.add_edge(article_name, link)

    node_counts = defaultdict(int)
    errors = 0
    all_nb_of_shortest_paths = []

    for _, row in pair_counts.iterrows():
        source = row['start_article']
        target = row['target_article']

        try:
            all_paths = list(nx.all_shortest_paths(G, source=source, target=target))
            nb_of_shortest_paths = len(all_paths)
            all_nb_of_shortest_paths.append(nb_of_shortest_paths)

            for path in all_paths:
                for node in path:
                    if node != source and node != target:
                        node_counts[node] += 1/nb_of_shortest_paths

        except nx.NetworkXNoPath:
            # Skip this source-target pair if no path exists
            errors += 1

    node_counts_df = pd.DataFrame(node_counts.items(), columns=['Node', 'ShortestPathCount'])
    node_counts_df = node_counts_df.sort_values(by='ShortestPathCount', ascending=False)

    return node_counts_df, errors, all_nb_of_shortest_paths