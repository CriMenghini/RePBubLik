import os
import sys
import math
import argparse
import itertools
from time import time
from collections import defaultdict

import pickle
import numpy as np
import networkx as nx
import multiprocessing as mp
from scipy.sparse import dia_matrix, diags

import parallel_walks 
import parallel_addition 
import parallel_centrality
from algo import Algorithms
from load_data import LoadData
from random_walks import chunk_random_walk
from utils import load_bubble_diameter, get_bad_and_good_nodes, get_centralities, sort_edges


parser = argparse.ArgumentParser(description='Run the entire/partial pipeline of FairRandomWalks experiments.')
parser.add_argument('-proc', type=str, default='radius', 
                    help='Part of pipeline to execute: diameter to compute the bubble diameter of all partitions, addition to compute new bubble diameter')
#parser.add_argument('-algo', type=str, default='baseline', help='If -proc is addition, choose the algo for addition')
parser.add_argument('-topic', type=str, help='Graph to analyze')
parser.add_argument('-t', type=int, default=15, help='Length of walks')
parser.add_argument('-b', type=int, default=2, help='Threshold to define good nodes')
parser.add_argument('-topk', type=int, default=10, help='Percentage of top-k central nodes to consider')
parser.add_argument('-maxedges', type=int, default=10, help='Maximum number of edges to add to the graph')
parser.add_argument('-iter', type=int, default=10, help='Number of iterations')
parser.add_argument('-unweighted', type=str, default='false', help='Weighted or unweighted graph')


#parser.add_argument('-topic', type=str, help='Graph to analyze')
args = parser.parse_args()

print(args.unweighted, type(args.unweighted), False == args.unweighted)
if args.unweighted == 'false':
    # Recall the graph of interest
    politics = LoadData(args.topic, uniform=False)
elif args.unweighted == 'true':
    politics = LoadData(args.topic, uniform=True)
else:
    print ('NO DATASET')
    exit()
id_color = politics.id_color
dictionary_weights = politics.dictionary_weights
id_name = politics.id_name
G = politics.G
edges_updated = set(list(G.edges()))

# Indeces for the adj matrix
node_id_matrix = {n:i for i,n in enumerate(list(G.nodes()))}
id_matrix_node = {j:i for i,j in node_id_matrix.items()}

# Define adj matrix
A = nx.adjacency_matrix(G)
d = diags(1/A.sum(axis=1).A.ravel())
A = A.T.dot(d).T


# Order nodes labels and colors
labels = np.array([j for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])
color_nodes = np.array([id_color[j] for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])

# Define RW parameters
delta = 0.05
eps = 1
r = int((args.t**2)/(eps**2)*np.log(1/delta))
print("Number of walks per node: ", r)

if args.proc == 'radius':
    if not os.path.exists(args.topic + '/'):
        os.makedirs(args.topic + '/')

    # Compute bubble diameter
    tt = time()
    result_chunks = parallel_walks.parallelization(A, labels, args.t, color_nodes, r)
    with open(args.topic + '/'  + args.topic + '_' + str(args.t) + '_bubble_diameters.pickle', 'wb') as f:
        pickle.dump(result_chunks, f)
    print(time()-tt)

    bubble_diameter = load_bubble_diameter(args.topic, id_matrix_node, args.t)
    red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices = get_bad_and_good_nodes(bubble_diameter, id_color, args.b, args.t)
    # Compute bad nodes centralities Blue
    nodes = np.array([node_id_matrix[n] for n in bad_blue_vertices])
    print('Number blue bad nodes: ', len(nodes))
    print("Number of walks per node: ", r)

    tt = time()
    result_chunks = parallel_centrality.parallelization(A, labels, args.t, color_nodes, r, nodes)
    with open(args.topic + '/'  + args.topic + '_' + 'blue' + '_' + str(args.t) + '_centralities.pickle', 'wb') as f:
        pickle.dump(result_chunks, f)
    print(time()-tt)

    # Compute bad nodes centralities Red
    nodes = np.array([node_id_matrix[n] for n in bad_red_vertices])
    print('Number red bad nodes: ', len(nodes))
    print("Number of walks per node: ", r)

    tt = time()
    result_chunks = parallel_centrality.parallelization(A, labels, args.t, color_nodes, r, nodes)
    with open(args.topic + '/'  + args.topic + '_' + 'red' + '_' + str(args.t) + '_centralities.pickle', 'wb') as f:
        pickle.dump(result_chunks, f)
    print(time()-tt)



elif args.proc == 'addition':
    bubble_diameter = load_bubble_diameter(args.topic, id_matrix_node, args.t)
    red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices = get_bad_and_good_nodes(bubble_diameter, id_color, args.b, args.t)
    
    # Get partitions' contribution
    s_red = np.sum(list(bad_red_vertices.values()))
    s_blue = np.sum(list(bad_blue_vertices.values()))

    perc_red = s_red/(s_red + s_blue)
    perc_blue = 1 - perc_red

    min_part = min(len(bad_red_vertices)*len(blue_bubble_diameter), len(bad_blue_vertices)*len(red_bubble_diameter))
    K_int = np.append([1],np.arange(2, min(min_part, args.maxedges), 2))
    
    K_R = [math.floor(i*perc_red) for i in K_int]
    K_B = [k - K_R[i] for i, k in enumerate(K_int)]


    for c in ['blue', 'red']:
        if c=='blue':
            bad = bad_blue_vertices
            K = K_B
        else:
            bad = bad_red_vertices
            K = K_R
        
        nodes = np.array([node_id_matrix[n] for n in bad])

        
        if c=='blue':
            alg = Algorithms(bad, red_bubble_diameter, labels, args.t, 
                        color_nodes, r, nodes, node_id_matrix, edges_updated, 0)
        else:
            alg = Algorithms(bad, blue_bubble_diameter, labels, args.t, 
                        color_nodes, r, nodes, node_id_matrix, edges_updated, 0)
        
        # ROV
        with open(args.topic + '/rov_candidate_edges.pickle','rb') as infile:
            possible_edges = pickle.load(infile)
        candidate_edges = [(i, j) for i,j,k in possible_edges]
        K = np.arange(2, min(len(candidate_edges), args.maxedges), 2)
        new_edges = candidate_edges
        alg._penalty_bulk_additions(A, K, new_edges, 
                                args.topic, algorithm='high_to_high',
                                color=c, perc=args.topk)
        
        
        # node2vec
        with open(args.topic + '/n2v_candidate_edges.pickle','rb') as infile:
            possible_edges = pickle.load(infile)

        candidate_edges = [(node_id_matrix[i], node_id_matrix[j]) for i,j,k in possible_edges]
        K = np.arange(2, min(len(candidate_edges), args.maxedges), 2)
        new_edges = candidate_edges
        alg._penalty_bulk_additions(A, K, new_edges, 
                                args.topic, algorithm='node2vec',
                                color=c, perc=args.topk)
        

        
        for ITER in range(int(args.iter)):
            if c == 'blue':
                alg = Algorithms(bad, red_bubble_diameter, labels, args.t, 
                                color_nodes, r, nodes, node_id_matrix, edges_updated, ITER)
            else:
                alg = Algorithms(bad, blue_bubble_diameter, labels, args.t, 
                                color_nodes, r, nodes, node_id_matrix, edges_updated, ITER)
            
            # Baseline
            candidate_edges = alg._compute_candidate_edge(algorithm='baseline', 
                                                        top_k=100, 
                                                        centrality=[])
            #K = np.arange(2, min(len(candidate_edges), args.maxedges), 2)
            alg._repeat_bulk_additions(A, K, candidate_edges, 
                                    args.topic, algorithm='baseline', 
                                    color=c, perc=100)
                                    
            # Recall centrality
            centrality = get_centralities(id_matrix_node, args.topic, c, args.t, old=False)
            perc = args.topk
            top_k = int(len(centrality)/100*perc)
            # Random pick central edges
            candidate_edges = alg._compute_candidate_edge('rand_central', top_k, centrality)
            alg._repeat_bulk_additions(A, K, candidate_edges, 
                                    args.topic, algorithm='rand_central', 
                                    color=c, perc=args.topk)
            # Random pick degree weighted central edges
            candidate_edges = alg._compute_candidate_edge('w_rand_central', top_k, centrality, G)
            alg._repeat_bulk_additions(A, K, candidate_edges, 
                                    args.topic, algorithm='w_rand_central', 
                                    color=c, perc=args.topk)
            
            # Pick sorted degree weighted central edges
            candidate_edges = alg._compute_candidate_edge('w_pen_central', top_k, centrality, G)
            new_edges = sort_edges(candidate_edges, K)
            alg._penalty_bulk_additions(A, K, new_edges, 
                                    args.topic, algorithm='w_pen_central', 
                                    color=c, perc=args.topk)
        