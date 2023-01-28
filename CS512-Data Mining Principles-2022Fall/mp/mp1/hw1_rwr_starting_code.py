import numpy as np
import networkx as nx
import math

G = nx.read_edgelist("email-Eu-core.txt")
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc)
A = nx.to_numpy_array(G)

"""
Normalize the given adjacency matrix
"""
D = np.zeros_like(A)
D_norm = np.zeros_like(D)
for i in range(A.shape[0]):
    D[i][i] = np.sum(A[i])
    D_norm[i][i] = 1/ math.sqrt(D[i][i])
A_hat = D_norm.dot(A).dot(D_norm)
# A_hat = D_norm * A * D_norm
print(A_hat)


"""
Implement random walk
"""
def run_rw(A, seed=0, eps=1e-6, max_iter=300):
    e = np.zeros(A.shape[0])
    e[seed] = 1
    r = e
    scores = []
    counter = 1
    while True:
        rt = A.dot(r)
        score = np.linalg.norm(rt - r)
        scores.append(score)
        if score <= eps:
            break
        r = rt
        counter += 1
        if counter >= max_iter:
            break
    return r, scores

r, scores = run_rw(A_hat, seed=0)
print("r:", r)
print("Scores:", scores)

"""
Implement random walk with restart
"""
def run_rwr(A, c, seed=0, eps=1e-6, max_iter=300):
    e = np.zeros(A.shape[0])
    e[seed] = 1
    r = e
    r1 = e
    scores = []
    counter = 1
    while True:
        r = c * (A.dot(r1)) + (1- c) * e
        score = np.linalg.norm(r - r1)
        scores.append(score)
        if score <= eps:
            break
        r1 = r
        counter += 1
        if counter >= max_iter:
            break
    return r, scores


r, scores = run_rwr(A_hat, c=0.9, seed=0)
print("r:", r)
print("Scores:", scores)