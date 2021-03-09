import os
import sys
import argparse
import itertools
from time import time
from collections import defaultdict

import pickle
import numpy as np
import networkx as nx
import multiprocessing as mp
from gensim.models import Word2Vec
from scipy.sparse import dia_matrix, diags

import parallel_entire_walks
from load_data import LoadData
from random_walks import chunk_random_walk
from utils import load_bubble_diameter, get_bad_and_good_nodes, get_centralities, sort_edges


parser = argparse.ArgumentParser(description='Compute node2vec embedding')
parser.add_argument('-proc', type=str, default='embedding', 
                    help='Part of pipeline to execute: diameter to compute the bubble diameter of all partitions, addition to compute new bubble diameter')
#parser.add_argument('-algo', type=str, default='baseline', help='If -proc is addition, choose the algo for addition')
parser.add_argument('-topic', type=str, help='Graph to analyze')
parser.add_argument('-t', type=int, default=15, help='Length of walks')
parser.add_argument('-r', type=int, default=10, help='Number of walks per node')


#parser.add_argument('-topic', type=str, help='Graph to analyze')
args = parser.parse_args()

# Recall the graph of interest
politics = LoadData(args.topic, uniform=False)
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

print("Number of walks per node: ", args.r)

if args.proc == 'embedding':
    # Compute bubble diameter
    tt = time()
    result_chunks = parallel_entire_walks.parallelization(A, labels, args.t, color_nodes, args.r)
    
    rc = [list(w) for nodes in result_chunks for w in nodes]
    print(rc[0], len(rc))
    workers = mp.cpu_count()
    
    model = Word2Vec(
                    rc,
                    size=128,
                    window=4,
                    min_count=0,
                    sg=1,
                    workers=workers,
                    iter=1,
                    )
    print(model.wv['0'])
    model.save(args.topic + '/' + args.topic + '_' + str(args.t) + "_word2vec.model")