from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx


class LoadData(object):
    
    def __init__(self, topic, uniform=True,
                 edges_file='clickstream_weighted_edges.tsv'):
        
        self.folder = 'data/{}/'.format(topic)
        self.uniform = uniform
        self.blue_nodes = pd.read_csv(self.folder + topic + '_blue_pre_labeled_nodes.tsv',
                                      sep='\t', 
                                      header=None)
        self.red_nodes = pd.read_csv(self.folder + topic + '_red_pre_labeled_nodes.tsv', 
                                     sep='\t', 
                                     header=None)
        self.id_name = self._mapping_id_name()
        self.id_color = self._mapping_id_color()
        assert len(self.id_name) == len(self.id_color)
        
        # Outdated edges, check this
        self.edges = self._weighted_edges(edges_file)
        self.color_edges = self._colored_edges()
        
        # Graph without sinks
        self.G = self._build_graph()
        self.red_nodes = [i for i in self.G.nodes() if self.id_color[i]=='red']
        self.blue_nodes = [i for i in self.G.nodes() if self.id_color[i]=='blue']
        
        self.dictionary_weights = self._different_weighted_neighbors()
        
    def _different_weighted_neighbors(self):
        """
        """
        
        dictionary_weights = defaultdict(dict)

        for n in self.G.nodes():
            neighbors = [i for i in self.G[n]]
            dictionary_weights[n]['neigh'] = neighbors
            if len(neighbors) != 0:
                #if uniform:
                weights = [1 for i in neighbors]
                normalized_weights = np.array(weights)/np.sum(weights)
                dictionary_weights[n]['uniform'] = normalized_weights
                # if weighted
                weights = [self.G[n][i]['weight'] for i in neighbors]
                normalized_weights = np.array(weights)/np.sum(weights)
                dictionary_weights[n]['weighted'] = normalized_weights

            else:
                dictionary_weights[n]['neigh'] = [n] + [i[0] for i in self.G.in_edges(n)]
                weights = [1 for i in dictionary_weights[n]['neigh']]
                normalized_weights = np.array(weights)/np.sum(weights)
                dictionary_weights[n]['uniform'] = normalized_weights
                
                weights = [self.G[i][n]['weight'] if i!=n else 10 for i in dictionary_weights[n]['neigh']]
                normalized_weights = np.array(weights)/np.sum(weights)
                dictionary_weights[n]['weighted'] = normalized_weights
        
        return dictionary_weights
            
            
    def _build_graph(self):
        """
        """
        
        G = nx.DiGraph()
        G.add_weighted_edges_from(self.edges)
        print(nx.info(G))

        # Remove sinks from the graph
        nodes_to_remove = [n for n in G.nodes() if len(G[n]) == 0]

        while len(nodes_to_remove) > 0:
            G.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = [n for n in G.nodes() if len(G[n]) == 0]
        #G.remove_nodes_from(nodes_to_remove)
        

        if self.uniform:
            for e in G.edges():
                G[e[0]][e[1]]['weight'] = 1

        print(nx.info(G))

        walked_edges = [(i,j,k) for i,j,k in self.edges if k>1]
        print("Fraction of walked edges: ", len(walked_edges)/len(self.edges))
        
        return G
        
    def _colored_edges(self):
        """
        """
        
        color_edges = {}

        for i,j,k in self.edges:
            if self.id_color[i] == 'red':
                if self.id_color[j] == 'red':
                    color_edges[(i,j)] = 'red_red'
                elif self.id_color[j] == 'blue':
                    color_edges[(i,j)] = 'red_blue'
            if self.id_color[i] == 'blue':
                if self.id_color[j] == 'red':
                    color_edges[(i,j)] = 'blue_red'
                elif self.id_color[j] == 'blue':
                    color_edges[(i,j)] = 'blue_blue'
                    
        return color_edges
        
        
    def _weighted_edges(self, edges_file):
        """
        """
        
        df_edges = pd.read_csv(self.folder + edges_file, sep='\t', header=None)
        print(df_edges)
        edges_multi = zip(list(df_edges[0].values), list(df_edges[1].values), list(df_edges[2].values))

        edges = list(filter(lambda x: (x[0] in self.id_name) and (x[1] in self.id_name),
                            edges_multi))

        return edges
        
    def _mapping_id_name(self):
        """
        
        """
        
        id_blue_nodes = list(self.blue_nodes[1].values)
        id_red_nodes = list(self.red_nodes[1].values)
        purple_nodes = set(id_blue_nodes).intersection(set(id_red_nodes))

        id_nodes = set(id_blue_nodes + id_red_nodes).difference(purple_nodes)
        
        id_name_blue = {i:j for i,j in zip(list(self.blue_nodes[1].values),\
                                           list(self.blue_nodes[0].values)) if i in id_nodes}
        id_name_red = {i:j for i,j in zip(list(self.red_nodes[1].values),\
                                          list(self.red_nodes[0].values)) if i in id_nodes}

        id_blue_nodes = list(id_name_blue.keys())
        id_red_nodes = list(id_name_red.keys())

        id_name = id_name_blue.copy()
        id_name.update(id_name_red)
        
        assert len(id_name_blue) + len(id_name_red) == len(id_nodes)
        #print(id_name)
        
        return id_name
    
    def _mapping_id_color(self):
        """
        """
        
        id_blue_nodes = list(self.blue_nodes[1].values)
        id_red_nodes = list(self.red_nodes[1].values)
        purple_nodes = set(id_blue_nodes).intersection(set(id_red_nodes))
        
        id_color = {}

        for i in id_blue_nodes:
            if i not in purple_nodes:
                id_color[i] = 'blue'
        for i in id_red_nodes:
            if i not in purple_nodes:
                id_color[i] = 'red'
                
        return id_color