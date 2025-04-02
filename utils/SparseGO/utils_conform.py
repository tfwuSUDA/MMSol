import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import numpy as np
import pandas as pd
import shutil
import scipy.stats as ss # for spearman_corr

def load_mapping(mapping_file):
    """
    Opens a txt file with two columns and saves the second column as the key of the dictionary and the first column as a value.

        Parameters
        ----------
        mapping_file: str, path to txt file

        Output
        ------
        mapping: dic

        Notes: used to read gene2ind.txt, drug2ind.txt

    """
    mapping = {} # dictionary of values on required txt

    file_handle = open(mapping_file) # function opens a file, and returns it as a file object.

    for line in file_handle:
        line = line.rstrip().split() # quitar espacios al final del string y luego separar cada elemento (en gene2ind hay dos elementos 3007	ZMYND8, los pone en una lista ['3007', 'ZMYND8'] )
        mapping[line[1]] = int(line[0]) # en gene2ind el nombre del gen es el key del dictionary y el indice el valor del diccionario

    file_handle.close()

    return mapping


def load_ontology(ontology_file, gene2id_mapping):
    """
    Creates the directed graph of the GO terms and stores the connected elements in arrays.

        Output
        ------
        dG: networkx.classes.digraph.DiGraph
            Directed graph of all terms

        terms_pairs: numpy.ndarray
            Store the connection between a term and a term

        genes_terms_pairs: numpy.ndarray
            Store the connection between a gene and a term
    """

    dG = nx.DiGraph() 

    file_handle = open(ontology_file) #  Open the file that has genes and go terms

    terms_pairs = [] # store the pairs between a term and a term  
    genes_terms_pairs = [] # store the pairs between a gene and a term  

    gene_set = set() # create a set (elements can't repeat)
    term_direct_gene_map = {}
    term_size_map = {}

    for line in file_handle:

        line = line.rstrip().split() # delete spaces and transform to list, line has 3 elements
        
        # No me hace falta el if, no tengo que separar las parejas
        if line[2] == 'default': # si el tercer elemento es default entonces se conectan los terms en el grafo
            dG.add_edge(line[0], line[1]) 
            terms_pairs.append([line[0], line[1]]) # Add the pair to the list
        else:
            if line[1] not in gene2id_mapping: 
                print(line[1])
                continue
            
            genes_terms_pairs.append([line[0], line[1]]) # add the pair

            if line[0] not in term_direct_gene_map: 
                term_direct_gene_map[ line[0] ] = set() # crea un set
 
            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

            gene_set.add(line[1]) 

    # terms_pairs: GO-GO
    # genes_terms_pairs: GO-gene
    terms_pairs = np.array(terms_pairs) # convert to 2d array
    genes_terms_pairs = np.array(genes_terms_pairs) # convert to 2d array

    file_handle.close()

    print('There are', len(gene_set), 'genes')

    for term in dG.nodes(): 

        term_gene_set = set() 

        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term] # genes conectados al term 

        deslist = nxadag.descendants(dG, term) 

        for child in deslist:
            if child in term_direct_gene_map: 
                term_gene_set = term_gene_set | term_direct_gene_map[child] # union of both sets, ahora tiene todos los genes los suyos y los de sus descendientes
        # if len(term_gene_set) == 0:
        #     print('There is empty terms, please delete term:', term)
        #     sys.exit(1)
        # else:
        #     # por ahora esta variable no me hace falta
        #     term_size_map[term] = len(term_gene_set) # cantidad de genes en ese term  (tomando en cuenta sus descendientes)
        term_size_map[term] = len(term_gene_set)


    leaves = [n for n in dG.nodes if dG.in_degree(n) == 0] # buscar la raiz
    #leaves = [n for n,d in dG.in_degree() if d==0]

    uG = dG.to_undirected() # Returns an undirected representation of the digraph
    connected_subG_list = list(nxacc.connected_components(uG)) 


    # Verify my graph makes sense...
    print('There are', len(leaves), 'roots:', leaves)  
    print('There are', len(dG.nodes()), 'terms')
    print('There are', len(connected_subG_list), 'connected components')

    # if len(leaves) > 1:
    #     print('There are more than 1 root of ontology. Please use only one root.')
    #     sys.exit(1)
    if len(connected_subG_list) > 1:
        print( 'There are more than connected components. Please connect them.')
        sys.exit(1)

    return dG, terms_pairs, genes_terms_pairs


def sort_pairs(genes_terms_pairs, terms_pairs, dG, gene2id_mapping):
    """
    Function concatenates the pairs and orders them, the parent term goes first.

        Output
        ------
        level_list: list
            Each array of the list stores the elements on a level of the hierarchy

        level_number: dict
            Has the gene and GO terms with their corresponding level number

        sorted_pairs: numpy.ndarray
            Contains the term-gene or term-term pairs with the parent element on the first column
    """

    all_pairs = np.concatenate((genes_terms_pairs,terms_pairs))  
    graph = dG.copy() #  Copy the graph to avoid modifying the original

    level_list = []   # level_list stores the elements on each level of the hierarchy
    level_list.append(list(gene2id_mapping.keys())) 

    while True:
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        if len(leaves) == 0:
            break
        
        level_list.append(leaves) # add the terms on each level
        graph.remove_nodes_from(leaves)

    
    level_number = {} # Has the gene and GO terms with their corresponding level number
    for i, layer in enumerate(level_list): 
        for _,item in enumerate(layer): 
            level_number[item] = i

    sorted_pairs = all_pairs.copy() # order pairs based on their level
    for i, pair in enumerate(sorted_pairs):
        level1 = level_number[pair[0]]
        level2 = level_number[pair[1]]
        if level2 > level1:  # the parent term goes first
            sorted_pairs[i][1] = all_pairs[i][0]
            sorted_pairs[i][0] = all_pairs[i][1]
    return sorted_pairs, level_list, level_number


def pairs_in_layers(sorted_pairs, level_list, level_number):
    """
    This function divides all the pairs of GO terms and genes by layers and adds the virtual nodes

        Output
        ------
        layer_connections: numpy.ndarray
            Contains the pairs that will be part of each layer of the model.
            Not all terms are connected to a term on the level above it. "Virtual nodes" are added to establish the connections between non-subsequent levels.

    """
    total_layers = len(level_list)-1 # Number of layers that the model will contain 总层级个数
    # Will contain the GO terms connections and gene-term connections by layers
    layer_connections = [[] for i in range(total_layers)]  

    for i, pair in enumerate(sorted_pairs):
        parent = level_number[pair[0]]  
        child = level_number[pair[1]] 

        # Add the pair to its corresponding layer
        layer_connections[child].append(pair) 

        # If the pair is not directly connected virtual nodes have to be added
        if child != 0:  
            virtual_node_layer = parent - 1
            while virtual_node_layer >= 1: 
                layer_connections[virtual_node_layer].append([pair[0],pair[0]]) 
                virtual_node_layer = virtual_node_layer-1

    # Delete pairs that are duplicated (added twice on the above step)
    for i,_ in enumerate(layer_connections):
        layer_connections[i] = np.array(layer_connections[i]) # change list to array
        layer_connections[i] = np.unique(layer_connections[i], axis=0)

    return layer_connections

