# from Project.include.article import Article
import sys
import os

# Add the base project directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print(sys.path)

from src.load_data import load_article_objects
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

articles = load_article_objects()

# Create an empty directed graph (since links have direction)
G = nx.DiGraph()

# Add nodes and edges
for article_name, article_obj in articles.items():
    # Add the article itself as a node
    G.add_node(article_name, category=article_obj.category)

    # Add edges for each link (out-edges)
    if article_obj.links:
        for link in article_obj.links:
            if link in articles:  # Ensure the linked article exists in the dataset
                G.add_edge(article_name, link)

# Now G is the NetworkX graph with articles as nodes and links as edges

# Example: print nodes and edges
# print("Nodes in the graph:")
# print(G.nodes(data=True))

# print("\nEdges in the graph:")
# print(G.edges())


# Choose a layout for the graph (spring layout is often a good default)
pos = nx.spring_layout(G, k=0.15, iterations=20)


'''
# Create the figure and plot the graph
plt.figure(figsize=(12, 12))  # Adjust the figure size based on the graph size

# Draw the nodes with labels and edges
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=8, font_weight="bold", edge_color="gray")

# Display the graph
plt.show()

net = Network(notebook=True, cdn_resources='remote', height="750px", width="100%")
print("created network")
net.from_nx(G)  # Load NetworkX graph into pyvis
print("loaded graph")
net.show("graph.html")  # This will open a zoomable, interactive graph in your browser
'''