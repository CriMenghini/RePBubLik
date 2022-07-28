import os
import sys
import copy
import random
import argparse
import itertools
from time import time
from collections import defaultdict

import pickle
import numpy as np
import networkx as nx
import multiprocessing as mp
from scipy.sparse import dia_matrix, diags
from scipy.sparse import csc_matrix, csr_matrix

import parallel_walks 
import parallel_addition 
import parallel_centrality
from algo import Algorithms
from load_data import LoadData
from random_walks import chunk_random_walk
from utils import load_bubble_diameter, get_bad_and_good_nodes, get_centralities, sort_edges


def get_tok_nodes(G, node_id_matrix, nodes, perc_k=5):
    """
    
    """
    
    red_degrees = {}
    for i in nodes:
        red_degrees[i] = G.degree(i)
    k = int(len(red_degrees)/100*perc_k)
    red_topk = [node_id_matrix[i] for i,j in sorted(red_degrees.items(), key=lambda x: x[1], reverse=True)][:k]

    return red_topk

def get_candidate_edges(G, red_nodes, blue_nodes, node_id_matrix, perc_k=5):
    """
    
    """
    
    red_topk = get_tok_nodes(G, node_id_matrix, red_nodes, perc_k = perc_k)
    blue_topk = get_tok_nodes(G, node_id_matrix, blue_nodes, perc_k = perc_k)
    
    candidate_edges = list(itertools.product(red_topk, blue_topk))
    # Add reverse edges since 
    candidate_edges = candidate_edges + [(j,i) for i,j in candidate_edges]
    
    return candidate_edges

def color_chain(A, node_id_matrix, red_nodes, alpha = 0.95):
    
    red_idx = [node_id_matrix[i] for i in red_nodes]
    
    r_red = np.zeros((len(node_id_matrix),1))
    r_red[red_idx] = 1/len(red_idx)
    r_red = csc_matrix(r_red)

    e_red = np.zeros((len(node_id_matrix),1))
    e_red[red_idx] = 1/len(red_idx)
    e_red = csc_matrix(e_red)

    old = r_red
    diff = 1
    i = 0
    while diff > 0.0001 or i < 200:
        new = alpha*A.dot(old) + (1-alpha)*e_red
        new = new/np.sum(new)
        diff = np.sum(np.abs(new - old))

        old = new
        i += 1

    r_red_final = new
    
    return r_red_final

def compute_controversy_score(A, node_id_matrix, topic_obj,red_topk, blue_topk,  perc=10, alpha=0.95):
    """
    
    """
    r_red_final = color_chain(csc_matrix(A), node_id_matrix, topic_obj.red_nodes, alpha=alpha)
    r_blue_final = color_chain(csc_matrix(A), node_id_matrix, topic_obj.blue_nodes, alpha=alpha)
    
    
    c_red = np.zeros((len(node_id_matrix),1))
    c_red[red_topk] = 1
    c_blue = np.zeros((len(node_id_matrix),1))
    c_blue[blue_topk] = 1
    
    diff_c = csr_matrix((c_red - c_blue).T)
    RWC = diff_c.dot(r_red_final - r_blue_final)
    
    return RWC[(0,0)]



def update_rwc(chunk, RWC, A, node_id_matrix, topic_obj, red_topk, blue_topk, perc=20, alpha=0.95):

    initial_RWC = RWC

    diff_RWC = {}
    i = 0
    for u,v in chunk:

        mat = copy.deepcopy(A)  
        new_prob = 1/(len(mat[:,u].indices)+1)
        mat[mat[:,u].indices, u] = new_prob
        mat[v, u] = new_prob  
        new = compute_controversy_score(mat, node_id_matrix, topic_obj, red_topk, blue_topk,perc, alpha)
        diff_RWC[(u,v)] = initial_RWC - new

        #print(i)
        i += 1
        
    return diff_RWC

def _get_chunks(num_nodes, list_nodes):
    """
    """
    
    n_proc = mp.cpu_count()
    max_nodes = num_nodes
    nodes = list_nodes[:max_nodes]
    if len(nodes) <= 40:
        n_proc = int(len(nodes)/1)
        n = int(len(nodes)/n_proc)
        return n_proc, [np.array(nodes[i:i + n]) for i in range(0, len(nodes), n)]
    
    n = int(len(nodes)/n_proc)
    
    chunks = [np.array(nodes[i:i + n]) for i in range(0, len(nodes), n)]
    
    return n_proc, chunks



def parallelization(RWC, A, node_id_matrix, topic_obj, red_topk, blue_topk, candidate_edges, perc=20, alpha=0.95):
        """
        """
        
        n_proc, chunks = _get_chunks(len(candidate_edges), candidate_edges)
        print('Number of cores: ', n_proc)
        #print('Number of chunks: ', len(chunks))

        with mp.Pool(processes=n_proc) as pool:
            proc_results = [pool.apply_async(update_rwc,
                                             args=(chunk, RWC, A, node_id_matrix, topic_obj, red_topk, blue_topk, perc, alpha, ))
                            for index_chunk, chunk in enumerate(chunks)]

            result_chunks = [r.get() for r in proc_results]#
        
        return result_chunks



parser = argparse.ArgumentParser(description='Run the entire/partial pipeline of FairRandomWalks experiments.')
parser.add_argument('-proc', type=str, default='radius', 
                    help='Part of pipeline to execute: diameter to compute the bubble diameter of all partitions, addition to compute new bubble diameter')
#parser.add_argument('-algo', type=str, default='baseline', help='If -proc is addition, choose the algo for addition')
parser.add_argument('-topic', type=str, help='Graph to analyze')
parser.add_argument('-unweighted', type=str, default='false', help='Weighted or unweighted graph')


#parser.add_argument('-topic', type=str, help='Graph to analyze')
args = parser.parse_args()



topic = args.topic

if args.unweighted == 'false':
    # Recall the graph of interest
    politics = LoadData(args.topic, uniform=False)
elif args.unweighted == 'true':
    politics = LoadData(args.topic, uniform=True)
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
# Column stochastic
A = A.T.dot(d)#.T

# Order nodes labels and colors
labels = np.array([j for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])
color_nodes = np.array([id_color[j] for i,j in sorted([(i,j) for i,j in id_matrix_node.items()], key=lambda x: x[0])])


red_topk = get_tok_nodes(G, node_id_matrix, politics.red_nodes, perc_k=10)
blue_topk = get_tok_nodes(G, node_id_matrix, politics.blue_nodes, perc_k=10)
RWC = compute_controversy_score(A, node_id_matrix, politics,red_topk, blue_topk,  perc=10, alpha=0.95)


candidate_edges = get_candidate_edges(G, politics.red_nodes, politics.blue_nodes, node_id_matrix, perc_k=10)#[:10]
candidate_edges = [(i,j) for i,j in candidate_edges if (id_matrix_node[i], id_matrix_node[j]) not in G.edges()]
#candidate_edges = [(i,j) for i,j in candidate_edges if id_color[id_matrix_node[i]] == c]
#print((len(candidate_edges)))

candidate_edges = random.sample(candidate_edges, 10000)
print('LEN CANDIDATES: ', len(candidate_edges))


candidate_to_save = []
result_chunks = parallelization(RWC, A, node_id_matrix, politics, red_topk, blue_topk, candidate_edges, perc=10, alpha=0.95)

tops = []
for i in range(len(result_chunks)):
    tops += sorted(result_chunks[i].items(), key=lambda x: x[1], reverse=True)
candidate_edges_ = sorted(tops, key=lambda x: x[1], reverse=True)[:min(len(tops), 2000)]#[0]


candidate_edges_ = [(i[0][0], i[0][1],1) for i in candidate_edges_]
with open('rov_candidate_edges.pickle', 'wb') as f:
    pickle.dump(candidate_edges_, f)
print(candidate_edges_)
