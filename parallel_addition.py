import numpy as np
import multiprocessing as mp

from random_walks import chunk_random_walk

def _get_chunks(num_nodes, list_nodes):
    """
    """
    
    n_proc = mp.cpu_count()
    max_nodes = num_nodes
    nodes = list_nodes[:max_nodes]
    n = int(len(nodes)/n_proc)
    
    chunks = [np.array(nodes[i:i + n]) for i in range(0, len(nodes), n)]
    
    return n_proc, chunks



def parallelization(A, labels, t, color_nodes, r, nodes):
        """
        """
        
        n_proc, chunks = _get_chunks(nodes.shape[0], list(nodes))
        print('Number of cores: ', n_proc, " NUMERO node ", len(list(nodes)))
        #print('Number of chunks: ', len(chunks))

        with mp.Pool(processes=n_proc) as pool:
            proc_results = [pool.apply_async(chunk_random_walk,
                                             args=(A, labels, t, color_nodes, chunk, r, index_chunk, ))
                            for index_chunk, chunk in enumerate(chunks)]

            result_chunks = [r.get() for r in proc_results]#
        
        return result_chunks