import numpy as np
import random
import matplotlib.pyplot as plt


# load network data 
network_file = "HW2_source/social_network.txt"
nodes = []
with open(network_file, "r") as f:
    for line in f:
        pair = [int(x) for x in line.strip().split(",")]
        nodes.append(pair)
adj_dim = np.max(nodes)+1
# setup adjacent matrix
adj_m = np.zeros((adj_dim, adj_dim))
network = {}
for node in nodes:
    adj_m[node[0]][node[1]] = 1
    adj_m[node[1]][node[0]] = 1
    if node[0] not in network:
        network[node[0]] = [node[1]]
    else:
        network[node[0]].append(node[1])
    if node[1] not in network:
        network[node[1]] = [node[0]]
    else:
        network[node[1]].append(node[0]) 


# compute eig value
# eig_w, eig_v = np.linalg.eig(adj_m)
# max_eig = np.max(eig_w)
# print("Max eigenvalue is:", max_eig)

def update_status(network, status, beta, delta):
    uninf_nodes = [x for x in status.keys() if status[x] == 0]
    inf_nodes = [x for x in status.keys() if status[x] == 1]
    next_status = {}
    # compute the prob of infection
    for node in uninf_nodes:
        adj_nodes = network[node]
        adj_nodes_status = [status[x] for x in adj_nodes]
        inf_p = 1-(1-beta)**np.sum(adj_nodes_status)
        # check infection or not
        if random.random() <= inf_p:
            next_status[node] = 1
        else:
            next_status[node] = 0 
    # compute the prob of recovery
    for node in inf_nodes:
        if random.random() <= delta:
            next_status[node] = 0
        else:
            next_status[node] = 1
    num_inf = np.sum(list(next_status.values()))
    num_uninf = len(next_status) - num_inf
    return next_status, num_inf, num_uninf

def draw_lineplot(infs, uninfs):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)  

    ax.plot(np.arange(len(infs)), infs, "-o", markersize=5, linewidth=2, color="r", label="Infected")
    ax.plot(np.arange(len(uninfs)), uninfs, "-o", markersize=5, linewidth=2, color="g", label="Healthy")
    ax.set_title("Number of infected nodes and healthy node at each time step")
    # ax.set_ylim([0.0, 2.3])
    # ax[i].yaxis.set_ticks(np.arange(0.85, 1.01, 0.05))
    ax.set_xticks(np.arange(0, len(infs), 5))
    plt.xlabel("Time step")
    plt.ylabel("Number") 
    ax.legend()
    plt.tight_layout()
    # plt.savefig("train_loss.png")
    plt.show()

# c
time_steps = 100
beta = 0.01
delta = 0.05
infs = []
uninfs = []
status = {x: 1 for x in range(adj_dim)}
for t in range(time_steps):
    status, inf_count, uninf_count = update_status(network, status, beta, delta)
    infs.append(inf_count)
    uninfs.append(uninf_count)
print("Infected number: ", infs)
print("Uninfected number: ", uninfs)
draw_lineplot(infs, uninfs)

# d
beta = 0.01
delta = 0.4
infs = []
uninfs = []
status = {x: 1 for x in range(adj_dim)}
for t in range(time_steps):
    status, inf_count, uninf_count = update_status(network, status, beta, delta)
    infs.append(inf_count)
    uninfs.append(uninf_count)
print("Infected number: ", infs)
print("Uninfected number: ", uninfs)
draw_lineplot(infs, uninfs)









