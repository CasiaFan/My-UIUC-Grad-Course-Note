import numpy as np
import networkx as nx

G = nx.read_edgelist("email-Eu-core.txt")
largest_cc = max(nx.connected_components(G), key=len)
largest_cc = list(largest_cc)
G = G.subgraph(largest_cc)
A = nx.to_numpy_array(G)

"""
Eigen decompose the given adjacency matrix
Conduct spectral partition
"""
D = np.diag(np.sum(A, axis=1))
L = D - A

w, v = np.linalg.eig(L)
w = w.real
v = v.real

idx = w.argsort()
v_2nd = v[:, idx[1]]

gnodes = np.array(G.nodes)
# gnodes = np.array(list(largest_cc))
part1_nodes = gnodes[v_2nd<=0]
part2_nodes = gnodes[v_2nd>0]

print("G1 nodes: {}".format(len(part1_nodes)))
print("G2 nodes: {}".format(len(part2_nodes)))


G1 = G.subgraph(part1_nodes)
G2 = G.subgraph(part2_nodes)

total_edges = len(G.edges)
G1_edges = len(G1.edges)
G2_edges = len(G2.edges)
print("G1 edges: {}".format(G1_edges))
print("G2 edges: {}".format(G2_edges))

print("Number of cut: {}".format(2*(total_edges-G1_edges-G2_edges)))




