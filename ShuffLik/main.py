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
from load_data import LoadData
from random_walks import chunk_random_walk
from utils import load_bubble_diameter, get_bad_and_good_nodes, get_centralities, sort_edges


parser = argparse.ArgumentParser(description='Run the entire/partial pipeline of FairRandomWalks experiments.')
parser.add_argument('-proc', type=str, default='diameter', 
                    help='Part of pipeline to execute: diameter to compute the bubble diameter of all partitions, addition to compute new bubble diameter')
#parser.add_argument('-algo', type=str, default='baseline', help='If -proc is addition, choose the algo for addition')
parser.add_argument('-topic', type=str, help='Graph to analyze')
parser.add_argument('-dataset', type=str, help='Data source')
parser.add_argument('-t', type=int, default=15, help='Length of walks')
parser.add_argument('-maxedges', type=int, default=10, help='Maximum number of edges to add to the graph')
parser.add_argument('-unweighted', type=str, default='false', help='Weighted or unweighted graph')
parser.add_argument('-edges', type=str, help='Diversity in recommendations?')
parser.add_argument('-centralities', type=str, help='Compute centralities?')


#parser.add_argument('-topic', type=str, help='Graph to analyze')
args = parser.parse_args()

if args.edges == 'diversity':
    kind_graph = 'clickstream_weighted_edges_diversity.tsv'
elif args.edges == 'vanilla':
    kind_graph = 'clickstream_weighted_edges.tsv'
elif args.edges == 'similarity':
    kind_graph = 'clickstream_weighted_edges_similarity.tsv'


print(args.unweighted, type(args.unweighted), False == args.unweighted)
if args.unweighted == 'false':
    # Recall the graph of interest
    politics = LoadData(args.topic, data=args.dataset, uniform=False, edges_file=kind_graph)
elif args.unweighted == 'true':
    politics = LoadData(args.topic, data=args.dataset, uniform=True, edges_file=kind_graph)
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

if args.proc == 'diameter':
    # Compute bubble diameter
    tt = time()
    if args.edges == 'vanilla':
        path = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + str(args.t) + '_bubble_diameters.pickle'
    elif args.edges == 'diversity':
        path = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + str(args.t) + '_bubble_diameters_diversity.pickle'
    elif args.edges == 'similarity':
        path = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + str(args.t) + '_bubble_diameters_similarity.pickle'
    
    result_chunks = parallel_walks.parallelization(A, labels, args.t, color_nodes, r)
    with open(path, 'wb') as f:
        pickle.dump(result_chunks, f)
    #print(time()-tt)

    if args.centralities == 'true':
        bubble_diameter = load_bubble_diameter(path, args.topic, id_matrix_node, args.t)
        # Set here b!!!
        red_br = {i:j for i,j in bubble_diameter.items() if id_color[i]=='red'}
        red_radius = list(red_br.values())
        b = np.median(red_radius)

        red_bubble_diameter, blue_bubble_diameter, bad_red_vertices, bad_blue_vertices, good_vertices = get_bad_and_good_nodes(bubble_diameter, id_color, b, args.t)
        
        
        
        # Compute bad nodes centralities Red
        nodes = np.array([node_id_matrix[n] for n in red_br])#np.array([node_id_matrix[n] for n in bad_red_vertices]) #
        print('Number red bad nodes: ', len(nodes))
        print("Number of walks per node: ", r)

        #tt = time()
        result_chunks = parallel_centrality.parallelization(A, labels, args.t, color_nodes, r, nodes)
        #print(result_chunks[0][-1])
        
        if args.edges == 'vanilla':
            path_centr = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + 'red' + '_' + str(args.t) + '_centralities.pickle'
        elif args.edges == 'diversity':
            path_centr = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + 'red' + '_' + str(args.t) + '_centralities_diversity.pickle'
        elif args.edges == 'similarity':
            path_centr = 'data/{}/{}/'.format(args.dataset,args.topic)  + args.topic + '_' + 'red' + '_' + str(args.t) + '_centralities_similarity.pickle'


        with open(path_centr, 'wb') as f:
            pickle.dump(result_chunks, f)
        print(time()-tt)