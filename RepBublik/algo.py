import copy
import random
import itertools
from time import time
from collections import defaultdict

import pickle
import numpy as np

import parallel_addition
from utils import get_candidate_edges



class Algorithms(object):
    
    def __init__(self, bad_vertices, good_vertices, labels, t,
                 color_nodes, r, nodes, node_id_matrix, edges, ITER,
                 G=None, id_color=None, bubble_diameter=None):
        """
        :bad_vertices: dict(node, diameter)
        :good_vertices: dict(node, diameter)
        :labels: ordered list of nodes
        :t: length of random walks 
        :color_nodes: ordered list of nodes colors
        :r: number of random walks for estimating the diameter
        :nodes: list ids of bad vertices
        :node_id_matrix: dict(node, adj matrix id)
        :edges_updated: graph edges
        :ITER:
        :G: graph
        :id_color: node to color
        :bubble_diameter:
        """

        self.bad_vertices = bad_vertices
        self.good_vertices = good_vertices
        self.labels = labels
        self.t = t
        self.color_nodes = color_nodes
        self.r = r
        self.nodes = nodes
        self.node_id_matrix = node_id_matrix
        self.edges = edges
        self.ITER = ITER
        self.G = G
        self.id_color = id_color
        self.bubble_diameter = bubble_diameter


    def _add_edges(self, to_add, mat, topic, algorithm, color, idx, perc=100):
        """Add edges and compute nuw diameter.

        :to_add: new edges. List of tuples
        :mat: adj matrix
        :topic: topic we study
        :algorithm: algo we consider
        :color: color of bad nodes
        :idx: iteration index
        """
        
        """
        # Uniform
        for i_out, i_in in to_add:
            new_prob = 1/(len(mat[i_out].indices)+1)
            mat[i_out, mat[i_out].indices] = new_prob
            mat[i_out, i_in] = new_prob
        """
        
        # Weighted
        for i_out, i_in in to_add:
            new_prob = 1/(len(mat[i_out].indices)+1)
            mat[i_out, mat[i_out].indices] = mat[i_out, mat[i_out].indices]*((len(mat[i_out].indices))/(len(mat[i_out].indices)+1))
            mat[i_out, i_in] = mat[i_out, i_in] + new_prob
            #print(np.sum(mat[i_out,:]))

        result_chunks = parallel_addition.parallelization(mat, self.labels, self.t, 
                                        self.color_nodes, self.r, self.nodes)

        # CHANGE FOR PCA
        with open(topic + '/'  + topic + '_' + color + '_' + algorithm + '_' + str(self.t) + '_perc_' + str(perc) + '_bubble_diameters_K_' + str(idx) + '_ITER_' + str(self.ITER) +'_no_parochial.pickle', 'wb') as f:
            pickle.dump(result_chunks, f)

        #with open(topic + '/'  + topic + '_' + color + '_' + algorithm + '_' + str(self.t) + '_perc_' + str(perc) + '_nodes_K_' + str(idx) + '_ITER_' + str(self.ITER) +'.pickle', 'wb') as f:
        #    if len(to_add) > 0:
        #        pickle.dump(list(zip(*to_add))[0], f)
        #    else:
        #        pickle.dump(' ', f)

        return mat


    def _compute_candidate_edge(self, algorithm, top_k=100, centrality=[], G=None, blue_nodes=None, red_nodes=None):
        """Get the list of possible edges

        :algorithm: algo we consider
        """

        if algorithm == 'baseline':
            # Define list of candidate edges
            candidate_edges = list(itertools.product(list(self.bad_vertices.keys()), 
                                                     list(self.good_vertices.keys())))     
        elif algorithm == 'rand_central':
            sorted_centralities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            candidate_edges = list(itertools.product([i for i,j in sorted_centralities[:top_k]], list(self.good_vertices.keys())))

        elif algorithm == 'w_rand_central':
            weighted_centrality = {i:j*(1/len(G[i])) for i,j in centrality.items()}
            weighted_sorted_centralities = sorted(weighted_centrality.items(), key=lambda x: x[1], reverse=True)
            candidate_edges = list(itertools.product([i for i,j in weighted_sorted_centralities[:top_k]], list(self.good_vertices.keys())))
        
        elif algorithm == 'w_pen_central':
            sorted_centralities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            other_color = list(self.good_vertices.keys())
            parochial = sorted_centralities

            # Check edges and give scores
            candidate_edges = sorted([(self.node_id_matrix[i],self.node_id_matrix[other_color[0]],k/len(G[i])) for (i,k) in parochial],
                                key=lambda x: x[2], reverse=True)

            return candidate_edges
        
        elif algorithm == 'high_to_high':
            candidate_edges = get_candidate_edges(G, blue_nodes, red_nodes, top_k)
        
        # Check edges are not in the graph    
        candidate_edges = [(self.node_id_matrix[i],self.node_id_matrix[j]) \
                              for i,j in candidate_edges if (i,j) not in self.edges]
        print("Number of candidate edges: ", len(candidate_edges))
        
        return candidate_edges




    def _repeat_bulk_additions(self, A, K, candidate_edges, topic, algorithm, color, perc=100):
        """Computer the new bubble diameter for each set of added edges.
        
        :A: adjacency matrix
        :K: list of number of total edges to add at each iteration
        :candidate_edges: list of edges one can add
        :algorithm: algo we consider
        """

        tt = time()
        to_add = []
        for idx, k in enumerate(K):
            if len(candidate_edges) >= k-len(to_add):
                pick = random.sample(candidate_edges, k-len(to_add))
            else:
                # salva in file la variabile to_add
                with open(topic + '/' + topic + '_' + color  + '_' + algorithm + '_added_edges_no_parochial.pickle', 'wb') as f:
                    pickle.dump(to_add, f)
                break
            to_add = to_add + random.sample(candidate_edges, k-len(to_add))
            candidate_edges = list(set(candidate_edges).difference(set(to_add)))
            mat = copy.deepcopy(A)  
            # Add edges and compute new bubble diameter      
            self._add_edges(to_add, mat, topic, algorithm, color, idx, perc)
            print(time()-tt)
        
        # salva in file la variabile to_add
        with open(topic + '/' + topic + '_' + color  + '_' + algorithm + '_added_edges_no_parochial.pickle', 'wb') as f:
            pickle.dump(to_add, f)

    def _penalty_bulk_additions(self, A, K, candidate_edges, topic, algorithm, color, perc=100):
        """Computer the new bubble diameter for each set of added edges with penality.
        """
        tt = time()
        to_add = []
        for idx, k in enumerate(K):
            print(k)
            to_add = candidate_edges[:k]
            mat = copy.deepcopy(A)  
            # Add edges and compute new bubble diameter      
            self._add_edges(to_add, mat, topic, algorithm, color, idx, perc)
            print(time()-tt)
        
        with open(topic + '/' + topic + '_' + color  + '_' + algorithm + '_added_edges_no_parochial.pickle', 'wb') as f:
            pickle.dump(to_add, f)

   