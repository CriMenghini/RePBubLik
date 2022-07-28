import os
import sys
import copy
import math
import pickle
import random
import argparse
import itertools
import numpy as np
from time import time
from collections import Counter, defaultdict

import networkx as nx
import multiprocessing as mp
from scipy.sparse import dia_matrix, diags, lil_matrix

import parallel_walks 
import parallel_addition 
import parallel_centrality
from load_data import LoadData
from random_walks import chunk_random_walk


parser = argparse.ArgumentParser(description='Swap edges.')
parser.add_argument('-topic', type=str, help='Graph to analyze')
parser.add_argument('-dataset', type=str, help='Data source')
parser.add_argument('-t', type=int, default=15, help='Length of walks')
parser.add_argument('-b', type=int, default=2, help='Threshold to define good nodes')
parser.add_argument('-maxswaps', type=int, default=10, help='Maximum number of swaps')
parser.add_argument('-iter', type=int, default=10, help='Number of iterations')
parser.add_argument('-centralities', type=str, help='Consider centralities?')
parser.add_argument('-edges', type=str, help='Diversity in recommendations?')

args = parser.parse_args()

if args.edges == 'diversity':
    kind_graph = 'clickstream_weighted_edges_diversity.tsv'
elif args.edges == 'vanilla':
    kind_graph = 'clickstream_weighted_edges.tsv'
elif args.edges == 'similarity':
    kind_graph = 'clickstream_weighted_edges_similarity.tsv'


politics = LoadData(args.topic, data=args.dataset, uniform=False, edges_file=kind_graph)
id_color = politics.id_color
dictionary_weights = politics.dictionary_weights
id_name = politics.id_name
G = politics.G
edges_updated = set(list(G.edges()))



# Indeces for the adj matrix
node_id_matrix = {n:i for i,n in enumerate(list(G.nodes()))}
id_matrix_node = {j:i for i,j in node_id_matrix.items()}


if args.edges == 'vanilla':
    path = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + str(args.t) + '_bubble_diameters.pickle'
elif args.edges == 'diversity':
    path = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + str(args.t) + '_bubble_diameters_diversity.pickle'
elif args.edges == 'similarity':
    path = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + str(args.t) + '_bubble_diameters_similarity.pickle'


with open(path, 'rb') as infile:
    data = pickle.load(infile)

bubble_diameter = {}
for k in range(len(data)):
    for i,j in zip(list(data[k][0]),list(data[k][1])):
        bubble_diameter[id_matrix_node[i]] = j[0]

#  BR per color
red_br = {i:j for i,j in bubble_diameter.items() if id_color[i]=='red'}
blue_br = {i:j for i,j in bubble_diameter.items() if id_color[i]=='blue'}

radiuses = list(bubble_diameter.values())
red_radius = list(red_br.values())
blue_radius = list(blue_br.values())

print ('INITIAL BR:{}'.format(np.mean(red_radius)))

# Set b
# Node polarity threshold
b = np.mean(red_radius)
m = np.max(red_radius)
B = m/2

# Get edges starting form red
red_src = [(src,trg) for src, trg in edges_updated if id_color[src]=='red']
# Take same color
red_to_red = [(src,trg) for src, trg in red_src if id_color[trg]=='red']
# Among those get the one with polarized target
polar_nodes = set([src for src,br in red_br.items() if br >= b])
polar_red_to_red = [(src, trg) for src, trg in red_to_red if trg in polar_nodes]
# Take different color
red_to_blue = [(src,trg) for src, trg in red_src if id_color[trg]=='blue']


if args.edges == 'vanilla':
    path_centr = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + 'red' + '_' + str(args.t) + '_centralities.pickle'
elif args.edges == 'diversity':
    path_centr = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + 'red' + '_' + str(args.t) + '_centralities_diversity.pickle'
elif args.edges == 'similarity':
    path_centr = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + 'red' + '_' + str(args.t) + '_centralities_similarity.pickle'



# Recall nodes' centralities
with open(path_centr, 'rb') as infile:
    hitting = pickle.load(infile)

hitting_time = defaultdict(dict)

for i in range(len(hitting)):
    index_nodes_bad = hitting[i][0]
    time_hit = hitting[i][1]
    hit_nodes = hitting[i][2]

    #avg_hit = np.mean(time_hit, axis=1)
    for h, n_h in enumerate(hit_nodes):
        for b, n_b in enumerate(index_nodes_bad):
            hitting_time[id_matrix_node[n_h]][id_matrix_node[n_b]] = time_hit[b,h]



#else:
#centrality = {i: args.t*len(hitting_time) - np.sum(list(j.values())) for i,j in hitting_time.items()}
centrality = {i: np.mean([args.t-l for k,l in j.items()]) for i,j in hitting_time.items()}

print('CHECK CENTRALITY ', len(centrality), len(red_br))

# Get list diversifying swaps
pol_edges_per_node = defaultdict(list)
for src, trg in polar_red_to_red:
    pol_edges_per_node[src] += [(src, trg, G[src][trg]['weight'])]
    
opposite_edges_per_node = defaultdict(list)
for src, trg in red_to_blue:
    opposite_edges_per_node[src] += [(src, trg, G[src][trg]['weight'])]

# Condition 1&2
pair_swappings = []
for r in red_br:
    try:
        a = pol_edges_per_node[r]
        b = opposite_edges_per_node[r]
        pair_swappings += list(itertools.product(a, b))
    except KeyError:
        continue

# Condition 3     
diversifying_swappings = []
for e1, e2 in pair_swappings:
    if e1[-1] > e2[-1]:
        #try:
        if args.centralities == 'true':
            diversifying_swappings += [(e1, e2, np.abs(e1[-1]-e2[-1])*centrality[e1[0]])]
        elif args.centralities == 'false':
            diversifying_swappings += [(e1, e2, np.abs(e1[-1]-e2[-1]))]
        #except KeyError:
        #    continue

print(len(pair_swappings), len(diversifying_swappings))

sort_swaps = sorted(diversifying_swappings, key=lambda x: x[-1], reverse=True)
print(sort_swaps[:10])


if args.edges == 'vanilla':
    save_swaps = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + str(args.t) + '_num_swaps.pickle'
elif args.edges == 'diversity':
    save_swaps = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + str(args.t) + '_num_swaps_diversity.pickle'
elif args.edges == 'similarity':
    save_swaps = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + str(args.t) + '_num_swaps_similarity.pickle'


with open(save_swaps, 'wb') as f:
    pickle.dump(len(sort_swaps), f)


# Swap K edges
k = min(len(sort_swaps), int(args.maxswaps))
print(k, len(sort_swaps), int(args.maxswaps))
swapped_edges = []

for n_swap in range(k):
    #print(n_swap)
    # select edge
    selected = sort_swaps[0]
    swapped_edges += [selected]
    
    # update
    upd_diversifying_swappings = []
    for e1, e2, diff in diversifying_swappings:
        if (e1[0] == selected[0][0]):
            if (e1[1] == selected[0][1]):
                if (e2[1] == selected[1][1]):
                    upd_diversifying_swappings += [((e1[0], e1[1], selected[1][2]), (e2[0], e2[1], selected[0][2]), diff)]
                else:
                    if args.centralities == 'true':
                        upd_diversifying_swappings += [((e1[0], e1[1], selected[1][2]), e2, np.abs(selected[1][2]-e2[-1])*centrality[e1[0]])]
                    elif args.centralities == 'false':
                        upd_diversifying_swappings += [((e1[0], e1[1], selected[1][2]), e2, np.abs(selected[1][2]-e2[-1]))]
            else:
                if (e2[1] == selected[1][1]):
                    if args.centralities == 'true':
                        upd_diversifying_swappings += [(e1, (e2[0], e2[1], selected[0][2]), np.abs(selected[0][2]-e1[-1])*centrality[e1[0]])]
                    elif args.centralities == 'false':
                        upd_diversifying_swappings += [(e1, (e2[0], e2[1], selected[0][2]), np.abs(selected[0][2]-e1[-1]))]

        else:
            upd_diversifying_swappings += [(e1, e2, diff)]
        

    # filter out those which do not hold property 3 anymore        
    diversifying_swappings = []
    for e1, e2, diff in upd_diversifying_swappings:
        if e1[-1] > e2[-1]:
            diversifying_swappings += [(e1, e2, diff)]

    
    
    print ('NEW LENGTH: ', len(diversifying_swappings), len(sort_swaps))
    sort_swaps = sorted(diversifying_swappings, key=lambda x: x[-1], reverse=True)
    if len(sort_swaps) == 0:
        break

print("Unique modified nodes:{}".format(len(set(list(zip(*list(zip(*swapped_edges))[0]))[0]))))

# Modify the transition matrix
swaps = [int(i) for i in np.linspace(0,k, 20)]
swaps_bis = swaps[1:]
edges_to_edit = list(zip(swaps[:-1], swaps_bis))

print(swapped_edges)

A = nx.adjacency_matrix(G)
d = diags(1/A.sum(axis=1).A.ravel())
A = A.T.dot(d).T
mat = copy.deepcopy(A)

# Order nodes labels and colors
labels = np.array([j for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])
color_nodes = np.array([id_color[j] for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])

# Define RW parameters
delta = 0.05
eps = 1
r = int((args.t**2)/(eps**2)*np.log(1/delta))
print("Number of walks per node: ", r)


for i,interval in enumerate(edges_to_edit):
    print(interval)
    for e1, e2, d in swapped_edges[interval[0]: interval[1]]:
        #print(e2[2], e1[2])
        mat[node_id_matrix[e1[0]], node_id_matrix[e1[1]]] = e2[2]
        mat[node_id_matrix[e2[0]], node_id_matrix[e2[1]]] = e1[2]
        


    if args.edges == 'vanilla':
        new_br = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_edit_{}_'.format(i) + str(args.t) + '_bubble_diameters'
    elif args.edges == 'diversity':
        new_br = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_edit_{}_'.format(i) + str(args.t) + '_bubble_diameters_diversity'
    elif args.edges == 'similarity':
        new_br = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_edit_{}_'.format(i) + str(args.t) + '_bubble_diameters_similarity'



    # Compute BR and change the 
    tt = time()
    if args.centralities == 'true':
        path_br = new_br + '.pickle'
    elif args.centralities == 'false':
        path_br = new_br + '_no_centr.pickle'
    
    result_chunks = parallel_walks.parallelization(mat, labels, args.t, color_nodes, r)
    with open(path_br, 'wb') as f:
        pickle.dump(result_chunks, f)
    print(time()-tt)