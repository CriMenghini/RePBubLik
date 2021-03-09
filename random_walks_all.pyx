from libc.stdlib cimport rand, RAND_MAX

import cython
cimport numpy as np
import numpy as np
from time import time
import copy 
import pickle
from time import time


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef Py_ssize_t _choose_one(double [:] pmf) nogil:
    """Random choice with discrete probabilities.
    Args:
        pmf (double[:]): probability mass function
    """
    cdef:
        Py_ssize_t i, length
        double random, total

    random = rand() / (RAND_MAX + 1.0)
    length = pmf.shape[0]
    i = 0
    total = 0.0

    while total < random and i < length:
        total += pmf[i]
        i += 1
    return i - 1

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def walk_random(normalized_csr, np.ndarray labels, int walk_length, 
                np.ndarray color_nodes, np.ndarray node):
    """Generate random walks for each node in a normalized sparse csr matrix.
    Args:
        normalized_csr (scipy.sparse.csr_matrix): normalized adjacency matrix
        labels (np.ndarray): array of node labels
        walk_length (int): length of walk
    Returns:
        np.array walks, np.array word frequencies
    """
    cdef:
        int [:] indices = normalized_csr.indices
        int [:] indptr = normalized_csr.indptr
        double [:] data = normalized_csr.data
        int num_nodes = len(node)

        int [:] neighbors
        double [:] weights
        int i, j, node_index, weight_index, next_index
        #str color = color_nodes[node[0]]
        #str color_next
        int count = 0

    walks = np.zeros([num_nodes, 1], dtype=object)
    #walks.fill(0)
    color = color_nodes[node[0]]
    
    for i in node:
        node_index = i
        
        for j in range(walk_length):
            neighbors = indices[indptr[node_index]:indptr[node_index+1]]
            weights = data[indptr[node_index]:indptr[node_index+1]]
            weight_index = _choose_one(weights)
            next_index = neighbors[weight_index]

            color_next = color_nodes[next_index]
                
            if color != color_next:
                walks[count][0] = walks[count][0] + 1
                break
            
            walks[count][0] = walks[count][0] + 1
            
            node_index = next_index
        
        count += 1

    return np.mean(walks) 


@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def chunk_random_walk(normalized_csr, np.ndarray labels, int walk_length, 
                      np.ndarray color_nodes, np.ndarray chunk, int r, int index_chunk):
    """Generate random walks for each node in a normalized sparse csr matrix.
    Args:
        normalized_csr (scipy.sparse.csr_matrix): normalized adjacency matrix
        labels (np.ndarray): array of node labels
        walk_length (int): length of walk
    Returns:
        np.array walks, np.array word frequencies
    """
    cdef:        
        int tot_nodes = chunk.shape[0]
        int i, n

    bubble_dimension = np.zeros([tot_nodes, 1], dtype=object)

    for i, n in enumerate(chunk):
        node = np.array([n]*r, dtype=int)

        node_diameter = walk_random(normalized_csr, labels, walk_length, color_nodes, node)
        bubble_dimension[i] = node_diameter
            
        #if i % 100:
        #    print(i)
        
    

    return (chunk, bubble_dimension)

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def multiple_nodes_walk_random(normalized_csr, np.ndarray labels, int walk_length, 
                               np.ndarray color_nodes, np.ndarray nodes, int r):
    """Generate random walks for each node in a normalized sparse csr matrix.
    Args:
        normalized_csr (scipy.sparse.csr_matrix): normalized adjacency matrix
        labels (np.ndarray): array of node labels
        walk_length (int): length of walk
    Returns:
        np.array walks, np.array word frequencies
    """
    cdef:        
        int tot_nodes = nodes.shape[0]
        int i, idx
        #np.ndarray node_diameter
        #np.ndarray bubble_dimension

    bubble_dimension = np.zeros([tot_nodes, 1], dtype=object)

    for idx, i in enumerate(nodes):
        node = np.array([i]*r, dtype=int)

        node_diameter = walk_random(normalized_csr, labels, walk_length, color_nodes, node)
        bubble_dimension[idx] = node_diameter

        #if idx % 100:
        #    print(idx)

    return bubble_dimension 


@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def new_diameters_bunch_edges(normalized_csr, np.ndarray labels, int walk_length, 
                               np.ndarray color_nodes, np.ndarray candidate_edges,
                             np.ndarray nodes, int r):
    """Generate random walks for each node in a normalized sparse csr matrix.
    Args:
        normalized_csr (scipy.sparse.csr_matrix): normalized adjacency matrix
        labels (np.ndarray): array of node labels
        walk_length (int): length of walk
    Returns:
        np.array walks, np.array word frequencies
    """
    cdef:        
        int tot_nodes = nodes.shape[0]
        int idx, i_out, i_in
        double new_prob, old_prob
        double tt
        #np.ndarray new_diameters
        #np.ndarray bubble_dimension

    

    new_diameters = np.zeros([len(candidate_edges), tot_nodes], dtype=object)#6000], dtype=object)#
    tt = time()
    for idx, (i_out, i_in) in enumerate(candidate_edges):#range(2500):
        A = copy.deepcopy(normalized_csr)  
        # Add edge
        new_prob = 1/(len(A[i_out].indices)+1)
        A[i_out, A[i_out].indices] = new_prob
        A[i_out, i_in] = new_prob

        bubble_dimension = multiple_nodes_walk_random(A, labels, walk_length, 
                                                      color_nodes, nodes, r)


        new_diameters[idx] = bubble_dimension.T

        #if idx % 100 == 0:
        print(idx, time()-tt)

    return candidate_edges, new_diameters


@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def centrality_walk_random(normalized_csr, np.ndarray labels, int walk_length, 
                np.ndarray color_nodes, np.ndarray node, int hit_node):
    """Generate random walks for each node in a normalized sparse csr matrix.
    Args:
        normalized_csr (scipy.sparse.csr_matrix): normalized adjacency matrix
        labels (np.ndarray): array of node labels
        walk_length (int): length of walk
    Returns:
        np.array walks, np.array word frequencies
    """
    cdef:
        int [:] indices = normalized_csr.indices
        int [:] indptr = normalized_csr.indptr
        double [:] data = normalized_csr.data
        int num_nodes = len(node)

        int [:] neighbors
        double [:] weights
        int i, j, node_index, weight_index, next_index
        #str color = color_nodes[node[0]]
        #str color_next
        int count = 0

    walks = np.zeros([num_nodes, 1], dtype=object)
    #walks.fill(0)
    #color = color_nodes[node[0]]
    
    for i in node:
        node_index = i
        
        for j in range(walk_length):
            neighbors = indices[indptr[node_index]:indptr[node_index+1]]
            weights = data[indptr[node_index]:indptr[node_index+1]]
            weight_index = _choose_one(weights)
            next_index = neighbors[weight_index]

            #color_next = color_nodes[next_index]
                
            if hit_node == next_index:
                walks[count][0] = walks[count][0] + 1
                break
            
            walks[count][0] = walks[count][0] + 1
            
            node_index = next_index
        
        count += 1

    return np.mean(walks) 


@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def centrality_chunk_random_walk(normalized_csr, np.ndarray labels, int walk_length, 
                      np.ndarray color_nodes, np.ndarray chunk, int r, int index_chunk,
                                np.ndarray hit_nodes):
    """Generate random walks for each node in a normalized sparse csr matrix.
    Args:
        normalized_csr (scipy.sparse.csr_matrix): normalized adjacency matrix
        labels (np.ndarray): array of node labels
        walk_length (int): length of walk
    Returns:
        np.array walks, np.array word frequencies
    """
    cdef:        
        int tot_nodes = chunk.shape[0]
        int tot_hitting_nodes = hit_nodes.shape[0]
        int i, n, idx

    bubble_dimension = np.zeros([tot_nodes, tot_hitting_nodes], dtype=object)

    for i, n in enumerate(chunk):
        node = np.array([n]*r, dtype=int)
        for idx, hit_node in enumerate(hit_nodes):
            node_diameter = centrality_walk_random(normalized_csr, labels, walk_length,
                                                   color_nodes, node, hit_node)
            
            bubble_dimension[i, idx] = node_diameter
            
        if i % 100:
            print(i)
        
    

    return (chunk, bubble_dimension, hit_nodes)


@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def walk_random_entire(normalized_csr, np.ndarray labels, int walk_length, 
                      np.ndarray color_nodes, np.ndarray node):
    """Generate random walks for each node in a normalized sparse csr matrix.
    Args:
        normalized_csr (scipy.sparse.csr_matrix): normalized adjacency matrix
        labels (np.ndarray): array of node labels
        walk_length (int): length of walk
    Returns:
        np.array walks, np.array word frequencies
    """
    cdef:
        int [:] indices = normalized_csr.indices
        int [:] indptr = normalized_csr.indptr
        double [:] data = normalized_csr.data
        int num_nodes = len(node)

        int [:] neighbors
        double [:] weights
        int i, j, node_index, weight_index, next_index
        #str color = color_nodes[node[0]]
        #str color_next
        int count = 0

    walks = np.zeros([num_nodes, walk_length + 1], dtype=object)
    
    #walks.fill(0)
    #color = color_nodes[node[0]]
    
    for i in node:
        node_index = i
        
        walks[count][0] = str(node_index)

        if node_index == 8:
            print('Aaaaaaaaa', walks[count][0])
        for j in range(1, walk_length + 1):
            neighbors = indices[indptr[node_index]:indptr[node_index+1]]
            weights = data[indptr[node_index]:indptr[node_index+1]]
            weight_index = _choose_one(weights)
            next_index = neighbors[weight_index]

            #color_next = color_nodes[next_index]
                
            #if color != color_next:
            #    walks[count][0] = walks[count][0] + 1
            #    break
            
            walks[count][j] = str(next_index)
            
            node_index = next_index
        
        count += 1

    return walks

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def chunk_random_walk_entire(normalized_csr, np.ndarray labels, int walk_length, 
                      np.ndarray color_nodes, np.ndarray chunk, int r, int index_chunk):
    """Generate random walks for each node in a normalized sparse csr matrix.
    Args:
        normalized_csr (scipy.sparse.csr_matrix): normalized adjacency matrix
        labels (np.ndarray): array of node labels
        walk_length (int): length of walk
    Returns:
        np.array walks, np.array word frequencies
    """
    cdef:        
        int tot_nodes = chunk.shape[0]
        int i, n

    bubble_dimension = np.zeros([tot_nodes*r, walk_length + 1], dtype=object)
    #print(bubble_dimension.shape)
    next_index = 0
    for i, n in enumerate(chunk):
        node = np.array([n]*r, dtype=int)
        if n == 8:
            print('AIOAIOAIO', next_index)
        node_walks = walk_random_entire(normalized_csr, labels, walk_length, color_nodes, node)
        bubble_dimension[next_index:next_index+r, :] = node_walks
        
        next_index = next_index + r
        #if i % 100:
        #    print(i)
        
    #print(node_walks, node_walks.shape)
    print(bubble_dimension.shape, bubble_dimension[240:240+r, :])
    return bubble_dimension