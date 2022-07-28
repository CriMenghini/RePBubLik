import os
import re
import pickle
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
from scipy.sparse import dia_matrix, diags, lil_matrix, csc_matrix, csr_matrix


from math import isclose
from collections import Counter
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from load_data import LoadData

def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0

parser = argparse.ArgumentParser(description='node2vec candidates.')
parser.add_argument('-topic', type=str, help='Graph to analyze')
parser.add_argument('-t', type=int, default=15, help='Length of walks')
parser.add_argument('-unweighted', type=str, default='false', help='Weighted or unweighted graph')

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


node_id_matrix = {n:i for i,n in enumerate(list(G.nodes()))}
id_matrix_node = {j:i for i,j in node_id_matrix.items()}


# Load BR
t = 10
with open(args.topic + '/' + args.topic + '_' + str(args.t) + '_bubble_diameters.pickle', 'rb') as infile:
    data = pickle.load(infile)

bubble_diameter = {}
for k in range(len(data)):
    for i,j in zip(list(data[k][0]),list(data[k][1])):
        bubble_diameter[id_matrix_node[i]] = j[0]

bad_diameters = {i:j for i,j in bubble_diameter.items() if j > args.t/2}

#Import node2vec model
model = Word2Vec.load("{}/{}_{}_word2vec.model".format(args.topic, args.topic, args.t))
print('Model loaded..')

node_list_1 = []
node_list_2 = []
for i,j in G.edges:
    node_list_1.append(node_id_matrix[i])
    node_list_2.append(node_id_matrix[j])
    
# combine all nodes in a list
node_list = node_list_1 + node_list_2
# remove duplicate items from the list
node_list = list(dict.fromkeys(node_list))
# build adjacency matrix
adj_G = nx.adjacency_matrix(G)


# Edges to learn
count = 0
to_learn = []

all_possible = np.array(range(adj_G.shape[1]))
for idx in bad_diameters:
    i = node_id_matrix[idx]
    sampling = np.array(random.sample(list(all_possible), 15))
    already_present = np.append(adj_G[i].indices, np.array([i]))
    new = set(list(sampling)).difference(set(list(sampling)).intersection(set(already_present)))
    
    to_learn += [(i,j) for j in new]
    count += 1
    if count%100==0:
        print(count)

count = 0
all_unconnected_pairs = []

all_possible = np.array(range(adj_G.shape[1]))
for idx in (set(G.nodes())).difference(set(bad_diameters)):
    i = node_id_matrix[idx]
    sampling = np.array(random.sample(list(all_possible), 10))
    already_present = np.append(adj_G[i].indices, np.array([i]))
    new = set(list(sampling)).difference(set(list(sampling)).intersection(set(already_present)))
    
    all_unconnected_pairs += [(i,j) for j in new]
    count += 1
    if count%100==0:
        print(count)

############################################################

node_1_tolearn = [i[0] for i in to_learn]
node_2_tolearn = [i[1] for i in to_learn]

data_tolearn = pd.DataFrame({'node_1':node_1_tolearn, 
                     'node_2':node_2_tolearn})

node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]

data = pd.DataFrame({'node_1':node_1_unlinked, 
                     'node_2':node_2_unlinked})

# add target variable 'link'
data['link'] = 0
fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})
# add the target variable 'link'
fb_df['link'] = 1

data = data.append(fb_df[['node_1', 'node_2', 'link']], ignore_index=True)

x = [operator_hadamard(model.wv[str(i)], model.wv[str(j)]) for i,j in zip(data['node_1'], data['node_2'])]
xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.3, 
                                                random_state = 35)

stats = data.link.value_counts()/len(data)
lr = LogisticRegression(class_weight={0:stats[0],1:stats[1]})

lr.fit(xtrain, ytrain)
predictions = lr.predict(xtest)

prediction_tolearn = np.array([])
for i in range(0, len(data_tolearn), 10000):
    print(i)
    x_to_learn = [operator_hadamard(model.wv[str(i)], model.wv[str(j)]) for i,j in zip(data_tolearn['node_1'][i:i+10000], data_tolearn['node_2'][i:i+10000])]
    prediction_tolearn = np.append(prediction_tolearn, lr.predict_proba(x_to_learn)[:,1])
    
data_tolearn['acceptance'] = prediction_tolearn
data_tolearn.sort_values('acceptance', ascending=False, inplace=True)
possible_edges = [(id_matrix_node[i],id_matrix_node[j],k) for i,j,k in zip(data_tolearn['node_1'], data_tolearn['node_2'], data_tolearn['acceptance'])]

with open(args.topic + '/n2v_candidate_edges.pickle', 'wb') as f:
    pickle.dump(possible_edges, f)