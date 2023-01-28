import numpy as np
import networkx as nx

G = nx.read_edgelist("email-Eu-core.txt")
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc)
A = nx.to_numpy_array(G)

"""
Normalize the given adjacency matrix
"""

"""
Implement random walk
"""

"""
Implement random walk with restart
"""
