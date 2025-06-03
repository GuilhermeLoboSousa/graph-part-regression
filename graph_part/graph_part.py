# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Union
import time
from collections import Counter
from itertools import product
import time

from tqdm import tqdm

from .transformations import TRANSFORMATIONS
from .train_val_test_split import train_val_test_split

from .needle_utils import get_len_dict, parse_fasta
import sys
sys.stdout.flush()

"""
This program partitions an entity set according to a single pairwise distance metric
and some desired threshold the partitions should fullfill.
The program seeks maximum distance between partitions. If the metric and desired 
threshold imply minimization you should designate the ('-tf', '--transformation')
parameter and specify either the one-minus or inverse transformation. The program 
converts the designated threshold the same way as the metric.
If for example a pair of entities have no distance between them (are identical)
when the metric is 1.0, and the desired threshold is 0.3, you would specify the threshold
-th 0.3 or --threshold 0.3, and then specify the desired transformation 
('-tf', '--transformation') such as -tf one-minus.
The entity/meta file can contain a priority designator and a label designator 
For example:
>entity_identifier1|exp=1|class=label1
>entity_identifier2|exp=0|class=label2
the priority name and label name should be designated using the appropriate parameters.
In this case you would specify the priority name -pn experimental and the label name 
-ln class
    
Author(s): Magnús Halldór Gíslason.
           Felix Teufel
           Developed in close collaboration with 
           José Juan Almagro Armenteros and Henrik Nielsen.
"""


def process_csv(line: str) -> Tuple[str, dict]:
    """ NOT IMPLEMENTED """
    raise NotImplementedError('Graph-Part does not support starting from .csv yet.')
    yield None, None

def process_fasta(line: str, priority_name:str, labels_name:str, labels: dict, seq_lens:dict) -> Tuple[str, dict]:
    """ Processes a fasta header lines or fasta meta data lines, if you will.
        Only supports interleaved fasta with > initialized header lines.
        Separate metadata with pipes | or colons :, whitespace between separators are ignored. 
        Some fasta files use the dash - as a separator. The dash is also used as an isoform indicator
        in accension numbers the dash separator. The option space dash space or [ - ] is implemented, but untested. """
    spl = line.strip().split('|')
    if ' - ' in spl[0]:
       spl = line.strip().split(' - ')
    if ':' in spl[0]:
        spl = line.strip().split(':')

    AC = spl[0].strip()[1:]
    priority = False
    label = '0'
    for s in spl[1:]:
        if '=' in s:
            param_spl = s.split('=')
            if param_spl[0] == priority_name:
                try:
                    priority = int(param_spl[1])==1
                except ValueError or TypeError:
                    raise TypeError("The input interpreted as priority designation did not conform as expected. Value interpeted: %r, Line: %r" % (param_spl[1], line))
            elif param_spl[0] == labels_name:
                label = str(param_spl[1].strip())

    if label not in labels:
        labels[label] = {'val':len(labels), 'num':0}
    labels[label]['num'] += 1

    label = labels[label]['val']

    node_data = {
        'priority': priority,
        'label-val': label,
        'length': seq_lens.get(AC, 0),
    }
    return AC, node_data


def load_entities(entity_fp: str, priority_name: str, labels_name: str):
    part_graph = nx.Graph()
    full_graph = nx.Graph()

    labels = {}
    ids,seqs=parse_fasta(entity_fp)
    seq_lens = get_len_dict(ids,seqs)

    with open(entity_fp) as inf:
        processing_as = None
        for line in inf:
            if '>' in line and processing_as != 'csv':
                AC, node_data = process_fasta(line, priority_name, labels_name, labels,seq_lens)
                processing_as = 'fasta'
            elif processing_as == 'fasta':
                continue
            else:
                AC, node_data = process_csv(line)
                processing_as = 'csv'

            full_graph.add_node(AC)
            nx.set_node_attributes(full_graph, {AC:node_data})
            
            part_graph.add_node(AC)

    return full_graph, part_graph, labels


from sklearn.preprocessing import KBinsDiscretizer

def calculate_global_score(cl_number, cluster_vector, target_vector, label_bins, n_partitions, n_bins, alpha, beta):
    """
    Calculates the global score for a given partition configuration.

    Parameters:
    -----------
    cl_number : np.ndarray
        Array indicating the partition assignment for each sample.
    cluster_vector : np.ndarray
        Array indicating the cluster each sample belongs to.
    target_vector : np.ndarray
        Array of continuous target values (e.g., regression targets).
    label_bins : np.ndarray
        Array of discretized target values (bins).
    n_partitions : int
        Number of partitions.
    n_bins : int
        Number of bins used for discretizing the target values.
    alpha : float
        Weight for the divergence in bin distribution.
    beta : float
        Weight for the imbalance in the sum of target values.

    Returns:
    --------
    total_score : float
        The global score for the current partition configuration.
    """
    # Initialize matrices to track bin distributions and target sums for each partition
    loc_bin_dist = np.zeros((n_partitions, n_bins))  # Bin distribution per partition
    loc_sum = np.zeros(n_partitions)  # Sum of target values per partition

    # Calculate bin distributions and target sums for each partition
    for p in range(n_partitions):
        # Get indices of samples assigned to partition `p`
        indices = np.where(cl_number == p)[0]
        
        # Calculate the sum of target values for partition `p`
        loc_sum[p] = np.sum(target_vector[indices])
        
        # Count the number of samples in each bin for partition `p`
        for bin_idx in label_bins[indices]:
            loc_bin_dist[p, bin_idx] += 1

    # Calculate the overall bin distribution across all partitions
    overall_bin_dist = np.sum(loc_bin_dist, axis=0) / np.sum(loc_bin_dist)

    # Initialize the total score
    total_score = 0

    # Calculate the score for each partition
    for p in range(n_partitions):
        # Normalize the bin distribution for partition `p`
        if np.sum(loc_bin_dist[p]) > 0:
            bin_dist_norm = loc_bin_dist[p] / np.sum(loc_bin_dist[p])
        else:
            bin_dist_norm = np.zeros(n_bins)

        # Calculate the L1 divergence between local and global bin distributions
        bin_div = np.sum(np.abs(bin_dist_norm - overall_bin_dist))

        # Calculate the imbalance in the sum of target values
        avg_sum = np.sum(loc_sum) / n_partitions  # Average sum of target values across partitions
        sum_diff = np.abs(loc_sum[p] - avg_sum)  # Absolute difference from the average

        # Combine the two components to calculate the score for partition `p`
        total_score += alpha * bin_div + beta * sum_diff

    return total_score
        
def optimize_partitions(cl_number, cluster_vector, target_vector, label_bins, n_partitions, n_bins, alpha, beta, n_mix):
    """
    Optimizes the partition assignments by performing random swaps and checking if the global score improves.

    Parameters:
    -----------
    cl_number : np.ndarray
        Array indicating the partition assignment for each sample.
    cluster_vector : np.ndarray
        Array indicating the cluster each sample belongs to.
    target_vector : np.ndarray
        Array of continuous target values (e.g., regression targets).
    label_bins : np.ndarray
        Array of discretized target values (bins).
    n_partitions : int
        Number of partitions.
    n_bins : int
        Number of bins used for discretizing the target values.
    alpha : float
        Weight for the divergence in bin distribution.
    beta : float
        Weight for the imbalance in the sum of target values.
    n_mix : int
        Number of random swaps to perform.

    Returns:
    --------
    best_cl_number : np.ndarray
        The optimized partition assignments.
    best_score : float
        The global score for the optimized partition configuration.
    """

    # Calculate the initial global score
    best_score = calculate_global_score(cl_number, cluster_vector, target_vector, label_bins, n_partitions, n_bins, alpha, beta)
    best_cl_number = np.copy(cl_number)

    # Perform random swaps to optimize the partition assignments
    for _ in range(n_mix):
        # Create a copy of the current partition assignments
        new_cl_number = np.copy(best_cl_number)

        # Select a random cluster and assign it to a new random partition
        random_cluster = np.random.choice(np.unique(cluster_vector))
        random_partition = np.random.choice(n_partitions)
        new_cl_number[cluster_vector == random_cluster] = random_partition

        # Calculate the new global score
        new_score = calculate_global_score(new_cl_number, cluster_vector, target_vector, label_bins, n_partitions, n_bins, alpha, beta)

        # If the new score is better, accept the change
        if new_score < best_score:
            best_score = new_score
            best_cl_number = new_cl_number

    return best_cl_number, best_score

def partition_data(full_graph: nx.classes.graph.Graph, 
                   part_graph: nx.classes.graph.Graph,
                   labels: dict,
                   threshold: float,
                   nr_of_parts: int,
                   mode: int):
    part_size = full_graph.number_of_nodes()//nr_of_parts

    label_limits = np.array([x[1]['lim'] for x in sorted(labels.items(), key=lambda x:x[1]['val'] )])
    print(part_size, label_limits)
    
    count = 0
    ## AC, cluster, label
    acs = []
    clusters = []
    labels = []
    print("Initialization mode", mode)

    ## Initialize the initialization
    for ind, AC in tqdm(enumerate(part_graph.nodes()), desc='Initializing'):
        label_counts = np.zeros(len(label_limits), dtype=int)
        label_counts[full_graph.nodes[AC]['label-val']] = 1
        cluster_nr = ind
        
        if mode == 'simple':
            d = full_graph.nodes[AC]
            acs.append(AC)
            clusters.append(cluster_nr)
            labels.append(d['label-val'])
        nx.set_node_attributes(part_graph, {
            AC:{
                'cluster': cluster_nr,
                'C-size': 1,
                'label-counts': label_counts
            }
        })

    ## Restricted closest neighbour linkage
    if mode in ['slow-nn', 'fast-nn']:
        ## Linking entities, if restrictions allow


        # NOTE sorting the networkx edges using sorted() becomes extremely slow on large graphs.
        # partly, because .edges takes forever to yield its EdgeView
        # workaround by instead extracting all the metric values into a numpy vector and argsorting this.
        # tested on ~400m edges, np.argsort 10 min vs. sorted() multiple hours
        queries = []
        libs = []
        metrics = []
                
        start = time.perf_counter()

        for qry, lib, data in full_graph.edges(data=True):
            queries.append(qry)
            libs.append(lib)
            metrics.append(data['metric'])

        elapsed_align = time.perf_counter() - start
        print(f"Edge iteration completed in {elapsed_align:0.2f} seconds.")
        
        
        metrics = np.array(metrics)
        inds = np.argsort(metrics)
        elapsed_align = time.perf_counter() - start
        print(f"Edge sorting competed at {elapsed_align:0.2f} seconds.")
        for idx in tqdm(inds, desc='clustering'):
            qry, lib = queries[idx], libs[idx]
            data = full_graph.edges[qry, lib]

            ### continue as below

        # for qry, lib, data in tqdm(sorted(full_graph.edges(data=True), key=lambda x: x[2]['metric']), desc='Clustering'):
            if data['metric'] > threshold:
                ## No need to continue if threshold reached.
                break
            
            if part_graph.has_edge(qry, lib):
                ## Update edge if it exists
                if part_graph[qry][lib]['metric'] > data['metric']:
                    nx.set_edge_attributes(part_graph,{(qry,lib):data['metric']}, 'metric')
                continue
            
            if part_graph.nodes[qry]['cluster'] == part_graph.nodes[lib]['cluster']:
               continue

            ## RESTRICTIONS!
            if part_graph.nodes[qry]['C-size'] >= part_size:
                continue
            if part_graph.nodes[lib]['C-size'] >= part_size:
                continue
            if (part_graph.nodes[qry]['label-counts']+part_graph.nodes[lib]['label-counts'] >= label_limits).any():
                continue
            
            ## Add edge and update mini-cluster
            attr = part_graph.nodes[qry]
            if attr['cluster'] != part_graph.nodes[lib]['cluster'] or mode == 'fast-nn':
                attr['C-size'] += part_graph.nodes[lib]['C-size']
                attr['label-counts'] += part_graph.nodes[lib]['label-counts']
            nx.set_node_attributes(part_graph, {qry:attr})
            part_graph.add_edge(qry, lib, metric=data['metric'])

            for descendant in nx.descendants(part_graph, qry):
                nx.set_node_attributes(part_graph, {descendant:attr})

            count += 1
            #if count % 10000 == 0:
            #    print("edges:", part_graph.number_of_edges()) 
            #    print (attr, data)
        #with open('graph_part_miniclusters.txt','w+') as outf:
        for cc_nr, cc in enumerate(nx.connected_components(part_graph)):
            for n in cc:
                d = full_graph.nodes[n]
                acs.append(n)
                clusters.append(cc_nr)
                labels.append(d['label-val'])
                    #outf.write("%s,%d\n" % (n,cc_nr))

    acs = np.array(acs)
    clusters = np.array(clusters)
    labels = np.array(labels)
    print(len(np.unique(labels)))
    partitioning = partition_assignment(clusters, labels, nr_of_parts, len(np.unique(labels)))
    for ind, p in enumerate(partitioning):
        attr = part_graph.nodes[acs[ind]]
        attr['cluster'] = p
        nx.set_node_attributes(part_graph,{acs[ind]:attr})

def remover( full_graph: nx.classes.graph.Graph, 
             part_graph: nx.classes.graph.Graph, 
             threshold:float, 
             json_dict: Dict[str, Any],
             move_to_most_neighbourly:bool = True, 
             ignore_priority:bool = True,
             verbose: bool = True ):
    
    if ignore_priority:
        json_dict['removal_step_1'] = {}
        dict_key = 'removal_step_1'
    else:
        json_dict['removal_step_2'] = {}
        dict_key = 'removal_step_2'
    
    if verbose:
        print("Min-threshold", "\t", "#Entities", "\t", "#Edges", "\t", "Connectivity", "\t", "#Problematics", "\t", "#Relocated", "\t", "#To-be-removed")

    removing_round = 0
    while True:
        between_connectivity = {}
        min_oc_wth = 1
        number_moved_from_train = 0
        number_moved_from_test = 0
        number_moved_from_val = 0

        cluster_0_number_proteins_in = Counter()
        cluster_1_number_proteins_in = Counter()
        cluster_2_number_proteins_in = Counter()

        nodes_by_cluster={0: {}, 1: {}, 2: {}}
        # first loop to know the number of proteins by range-size(10 in 10) and label
        for n,d in full_graph.nodes(data=True):
            cluster = part_graph.nodes[n]['cluster']
            seq_length = d['length']
            range_key = (seq_length // 10) * 10
            label_val = d['label-val']

            if cluster == 0:
                cluster_0_number_proteins_in[(range_key, label_val)] += 1
            elif cluster == 1:
                cluster_1_number_proteins_in[(range_key, label_val)] += 1
            elif cluster == 2:
                cluster_2_number_proteins_in[(range_key, label_val)] += 1
            if (range_key, label_val) not in nodes_by_cluster[cluster]:
                nodes_by_cluster[cluster][(range_key, label_val)] = []
            nodes_by_cluster[cluster][(range_key, label_val)].append(n)

        #the goal is to have more proteins in cluster 0 than in cluster 1 for the same range-size and label
        for (range_key, label_val), count in cluster_1_number_proteins_in.items():
            #ensure that if a range do not have any protein in cluster 0, it is created- to avoid key error
            if (range_key, label_val) not in cluster_0_number_proteins_in:
                cluster_0_number_proteins_in[(range_key, label_val)] = 0
                nodes_by_cluster[0][(range_key, label_val)] = []
            if cluster_0_number_proteins_in[(range_key, label_val)] < count:#if the number of proteins in cluster 0 is less than the number of proteins in cluster 1
                nodes_cluster_1 = nodes_by_cluster[1][(range_key, label_val)]#get all the proteins in cluster 1 (test)
                nodes_cluster_0 = nodes_by_cluster[0][(range_key, label_val)]#get all the proteins in cluster 0 (train)

                # Move all proteins from cluster 1 to cluster 0, to ensure that the number of proteins in cluster 0 is greater than the number of proteins in cluster 1 for the same range-size and label
                for node in nodes_cluster_1:
                    part_graph.nodes[node]['cluster'] = 0
                    cluster_0_number_proteins_in[(range_key, label_val)] += 1
                    cluster_1_number_proteins_in[(range_key, label_val)] -= 1

                # move all proteins from cluster 0 to cluster 1
                for node in nodes_cluster_0:
                    part_graph.nodes[node]['cluster'] = 1
                    cluster_1_number_proteins_in[(range_key, label_val)] += 1
                    cluster_0_number_proteins_in[(range_key, label_val)] -= 1

        for n, d in full_graph.nodes(data=True):
            neighbours = nx.neighbors(full_graph, n)
            possible_neighbours= [n for n in neighbours]
            neighbour_clusters = Counter((part_graph.nodes[nb]['cluster'] for nb in possible_neighbours if full_graph[n][nb]['metric'] <threshold))
            cluster = part_graph.nodes[n]['cluster']
            nb_oc_wth = []
            seq_length = d['length']
            range_key = (seq_length // 10) * 10
            label_val = d['label-val']

            if move_to_most_neighbourly:
                if len(neighbour_clusters) > 0: # if there are neighbours (with metric<treshold- similar sequences), move to the cluster with the most neighbours to ensure no bias in sets
                    most_neighbourly_cluster =  max(neighbour_clusters, key=neighbour_clusters.get) #sequence is moved to the cluster with the most neighbours
                    if cluster == 0 and most_neighbourly_cluster in [1, 2]: #if the sequence is in cluster 0 and the most neighbourly cluster is 1 or 2 (move from train to test or val)
                        part_graph.nodes[n]['cluster'] = most_neighbourly_cluster
                        cluster = part_graph.nodes[n]['cluster']
                        number_moved_from_train += 1
                    elif cluster == 1 and most_neighbourly_cluster == 2: #if the sequence is in cluster 1 and the most neighbourly cluster is 2 (move from test to val)
                        part_graph.nodes[n]['cluster'] = most_neighbourly_cluster
                        cluster = part_graph.nodes[n]['cluster']
                        number_moved_from_test += 1
                    elif cluster == 2 and most_neighbourly_cluster == 1: #if the sequence is in cluster 2 and the most neighbourly cluster is 1 (move from val to test)
                        part_graph.nodes[n]['cluster'] = most_neighbourly_cluster
                        cluster = part_graph.nodes[n]['cluster']
                        number_moved_from_val += 1

            if ignore_priority:# and full_graph.nodes[n]['priority']:
                between_connectivity[n] = 0
                continue

            for neighbour in possible_neighbours: #strategy began to eliminate the most problematic entities/sequences that have connections with other sequences in other clusters (below metric)
                nb_cluster = part_graph.nodes[neighbour]['cluster']
                if nb_cluster != cluster and full_graph[n][neighbour]['metric'] < 0.6:
                    min_oc_wth = min(min_oc_wth, full_graph[n][neighbour]['metric'])
                    nb_oc_wth.append(full_graph[n][neighbour]['metric'])
            
            between_connectivity[n] = len(nb_oc_wth)
        nx.set_node_attributes(full_graph, between_connectivity, 'between_connectivity')
        bc_sum = np.sum(np.fromiter((d['between_connectivity'] for n,d in full_graph.nodes(data=True)),int))
        bc_count = np.sum(np.fromiter((1 for n,d in full_graph.nodes(data=True) if d['between_connectivity'] > 0),int)) #know the sequences that have more problematics neighbours
        removing_round += 1
        number_to_remove = int(bc_count*np.log10(removing_round)/100)+1 # int(bc_count*0.01)+1 (another strategy) #gradualy increase the number of sequences to remove
        moved = number_moved_from_train + number_moved_from_test + number_moved_from_val
        remove_these = [x[0] for x in sorted(((n,d['between_connectivity']) for n,d in full_graph.nodes(data=True) if d['between_connectivity'] > 0), key=lambda x:x[1], reverse=True)[:number_to_remove]]

        if verbose:
            print(round(min_oc_wth,7), "\t\t", full_graph.number_of_nodes(), "\t\t", full_graph.number_of_edges(), "\t\t", bc_sum, "\t\t", bc_count, "\t\t", moved, "\t\t", len(remove_these))
        
        json_dict[dict_key][removing_round] = {
                                                "Min-threshold": round(min_oc_wth,7) ,
                                                "#Entities": full_graph.number_of_nodes(),
                                                "#Edges": full_graph.number_of_edges(),
                                                "Connectivity": int(bc_sum), #soma de todos os problemas possiveis
                                                "#Problematics": int(bc_count), # nº de nos que tem conexoes com outros nos maximo n de seqs
                                                "#Relocated": moved, 
                                                "#To-be-removed":len(remove_these)
                                                }
        full_graph.remove_nodes_from(remove_these)
        
        # If we've removed the last problematic entities, we stop
        if full_graph.number_of_nodes()==0 or bc_sum==0 or len(remove_these) == 0:
            break

def score_partitioning(df:pd.core.frame.DataFrame) -> float:
    s0 = df.shape[0]
    s1 = df.shape[1]
    return float((df.product(axis=1)**(1/s1)).product()**(1/s0))

def display_results(
    part_graph: nx.classes.graph.Graph, 
    full_graph: nx.classes.graph.Graph,
    labels: dict,
    nr_of_parts: int,
    verbose: bool = True) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    """ """
    df = pd.DataFrame(((d) for n,d in full_graph.nodes(data=True)))
    df['cluster'] = [part_graph.nodes[n]['cluster'] for n in full_graph.nodes()]

    # It can happen that removal completely removed one partition.
    # In this case, we need to report back an error
    if len(df['cluster'].unique()) < nr_of_parts:
        error_string = f'''
        Impossible to generate the desired {nr_of_parts} partitions at the current partitioning threshold.
        Removal of sequences to achieve separation results in loss of {nr_of_parts-len(df['cluster'].unique())} complete partitions.
        '''
        raise RuntimeError(error_string)

    df['AC'] = [n for n in full_graph.nodes()]
    result = df.groupby(['cluster','label-val'])['AC'].count().reset_index().pivot_table(values='AC',columns=['label-val'],index=['cluster']).T
    df.set_index('AC', inplace=True)
    result['label'] = ''
    for l in labels:
        result.loc[labels[l]['val'], 'label'] = l
        result['mean'] = result[list(range(nr_of_parts))].mean(axis=1)
        result['count'] = result[list(range(nr_of_parts))].sum(axis=1)
    
    if verbose:
        print(result)
        print()
        print("Partitioning score:", score_partitioning(result[range(nr_of_parts)]))
        print()
    return df, result
    

def removal_needed(
    part_graph: nx.classes.graph.Graph, 
    full_graph: nx.classes.graph.Graph,
    threshold: float) -> bool:
    """ """
    return True


def make_graphs_from_sequences(config: Dict[str, Any], threshold: float, json_dict: Dict[str,Any], verbose: bool = True) -> Tuple[nx.classes.graph.Graph, nx.classes.graph.Graph, dict]:
    '''
    This function performs the alignments and constructs the graphs.

    Parameters:
    ------------
        config: dict
            A dictionary of all parameters needed to run graph-part

        threshold: float
            The threshold to use for partitioning. Alignments 
            are discarded if their distance is above the threshold.

        json_dict: dict
            A dictionary to collect outputs for the final report.

        verbose:  bool
            If True, print all processing steps to command line.

    Returns:
    ------------
        full_graph: nx.classes.graph.Graph
            Networkx graph that has sequences as nodes and their distances as edge attributes
        part_graph: nx.classes.graph.Graph
            Networkx graph that collects the final partition assignments.
        labels: dict
            Dictionary of label statistics
    '''
    full_graph, part_graph, labels = load_entities(config['fasta_file'], config['priority_name'], config['labels_name'])

    for l in labels:
        """ Find the expected number of entities labelled l in any partition """
        labels[l]['lim'] = labels[l]['num']//config['partitions']

    ## Let's see the initial label distribution
    if verbose:
        print(pd.DataFrame(labels).T)
    json_dict['labels_start'] = labels


    if config['alignment_mode'] == 'precomputed':
        from .precomputed_utils import load_edge_list
        print('Parsing edge list.')
        load_edge_list(config['edge_file'], full_graph, config['transformation'], threshold, config['metric_column'])
        elapsed_align = time.perf_counter() - json_dict['time_script_start'] 
        if verbose:
            print(f"Edge list parsing executed in {elapsed_align:0.2f} seconds.")

    elif config['alignment_mode'] == 'mmseqs2':
        from .mmseqs_utils import generate_edges_mmseqs
        generate_edges_mmseqs(config['fasta_file'], full_graph, config['transformation'], threshold, config['threshold'], denominator=config['denominator'], delimiter='|', is_nucleotide=config['nucleotide'], use_prefilter=config['prefilter'])
        elapsed_align = time.perf_counter() - json_dict['time_script_start'] 
        if verbose:
            print(f"Pairwise alignment executed in {elapsed_align:0.2f} seconds.")    

    elif config['alignment_mode'] == 'needle' and config['threads']>1:
        from .needle_utils import generate_edges_mp
        print('Computing pairwise sequence identities.')
        generate_edges_mp(config['fasta_file'], full_graph, config['transformation'], threshold, denominator=config['denominator'], n_chunks=config['chunks'], n_procs=config['threads'], parallel_mode=config['parallel_mode'], triangular=config['triangular'], delimiter='|', 
                            is_nucleotide=config['nucleotide'], gapopen=config['gapopen'], gapextend=config['gapextend'], endweight=config['endweight'], endopen=config['endopen'], endextend=config['endextend'], matrix=config['matrix'])
        elapsed_align = time.perf_counter() - json_dict['time_script_start'] 
        if verbose:
            print(f"Pairwise alignment executed in {elapsed_align:0.2f} seconds.")

    elif config['alignment_mode'] == 'needle':
        from .needle_utils import generate_edges
        print('Computing pairwise sequence identities.')
        generate_edges(config['fasta_file'],full_graph, config['transformation'], threshold, denominator=config['denominator'], delimiter='|',
                            is_nucleotide=config['nucleotide'], gapopen=config['gapopen'], gapextend=config['gapextend'], endweight=config['endweight'], endopen=config['endopen'], endextend=config['endextend'], matrix=config['matrix'])
        elapsed_align = time.perf_counter() - json_dict['time_script_start'] 
        if verbose:
            print(f"Pairwise alignment executed in {elapsed_align:0.2f} seconds.")

    elif config['alignment_mode'] == 'mmseqs2needle':
        from .mmseqs_needle_combined_utils import generate_edges_mmseqs_needle_combined
        recompute_threshold = TRANSFORMATIONS[config['transformation']](config['recompute_threshold'])
        print('Computing pairwise sequence identities.')
        generate_edges_mmseqs_needle_combined(
            config['fasta_file'],
            full_graph,
            transformation=config['transformation'],
            threshold=threshold,
            recompute_threshold=recompute_threshold,
            denominator_needle=config['denominator_needle'],
            denominator_mmseqs=config['denominator_mmseqs'],
            n_procs=config['threads'],
            parallel_mode=config['parallel_mode'],
            triangular=config['triangular'],
            delimiter='|',
            is_nucleotide=config['nucleotide'],
            use_prefilter=config['prefilter'],
            gapopen=config['gapopen'],
            gapextend=config['gapextend'],
            endweight=config['endweight'],
            endopen=config['endopen'],
            endextend=config['endextend'],
            matrix=config['matrix']
        )
        # generate_edges_mmseqs_needle_combined(config['fasta_file'], full_graph, config['transformation'], threshold, recompute_threshold, config['threshold'], denominator_needle=config['denominator_needle'], denominator_mmseqs=config['denominator_mmseqs'], n_procs=config['threads'], parallel_mode=config['parallel_mode'], triangular=config['triangular'], delimiter='|', 
                                            #   is_nucleotide=config['nucleotide'], use_prefilter=config['prefilter'], gapopen=config['gapopen'], gapextend=config['gapextend'], endweight=config['endweight'], endopen=config['endopen'], endextend=config['endextend'], matrix=config['matrix'])

    else:
        raise NotImplementedError('Encountered unspecified alignment mode. This should never happen.')

    
    return full_graph, part_graph, labels


def partition_and_remove(full_graph: nx.classes.graph.Graph, part_graph: nx.classes.graph.Graph, labels: dict, json_dict: dict,
                            threshold: float, config: dict, write_intermediate_file: bool = False, verbose: bool = True) -> pd.core.frame.DataFrame:
    '''
    This function runs the core Graph-Part algorithm. Its inputs are generated by
    `make_graphs_from_sequences` or another function that produces outputs of the same
    kind for non-sequence data.
    '''
    
    partition_data(full_graph, part_graph, labels, threshold, config['partitions'], config['initialization_mode'])

    df, result = display_results(part_graph, full_graph, labels, config['partitions'], verbose=verbose)
    if config['test_ratio']>0:
        train_val_test_split(part_graph, full_graph, threshold, config['test_ratio'], config['val_ratio'], config['partitions'])
        config['partitions'] = 3 if config['val_ratio']>0 else 2

    df, result = display_results(part_graph, full_graph, labels, config['partitions'], verbose=verbose)
    if write_intermediate_file:
        df.to_csv(config['out_file'] + "pre-removal")
    print('Currently have this many samples:', full_graph.number_of_nodes())

    json_dict['partitioning_pre_removal'] = result.to_json()
    json_dict['samples_pre_removal'] = full_graph.number_of_nodes()
    json_dict['score_pre_removal'] = score_partitioning(result[range(config['partitions'])])

    
    ## Check if we need to remove any
    if removal_needed(part_graph, full_graph, threshold):     
        print('Need to remove! Currently have this many samples:', full_graph.number_of_nodes())

        remover(full_graph, part_graph, threshold, json_dict, config['allow_moving'], True, verbose=verbose)    

    if removal_needed(part_graph, full_graph, threshold):   
        print('Need to remove priority! Currently have this many samples:', full_graph.number_of_nodes())
        remover(full_graph, part_graph, threshold, json_dict, config['allow_moving'], False, verbose=verbose)    

    print('After removal we have this many samples:', full_graph.number_of_nodes())


    df, result = display_results(part_graph, full_graph, labels, config['partitions'], verbose=verbose)

    json_dict['partitioning_after_removal'] = result.to_json()
    json_dict['samples_after_removal'] = full_graph.number_of_nodes()
    json_dict['score_after_removal'] = score_partitioning(result[range(config['partitions'])])

    if removal_needed(part_graph, full_graph, threshold):
        print ("Something is wrong! Removal still needed!")
        json_dict['removal_needed_end'] = True
    else:
        json_dict['removal_needed_end'] = False

    return df


def run_partitioning(config: Dict[str, Union[str,int,float,bool]], write_output_file: bool = True, write_json_report: bool=True, verbose: bool=True) -> pd.core.frame.DataFrame:
    '''
    Core Graph-Part partitioning function. `config` contains all parameters passed from the command line
    or Python API. See `cli.py` for the definitions.  

    Parameters:
    -----------
    config:
        A dictionary of all parameters needed to run graph-part
    write_output_file: bool
        If True, write final assignment table to disk.
    write_json_report: bool
        If True, write a report of all summary statistics. Used by the webserver.
    verbose:  bool
        If True, print all processing steps to command line.
    '''

    s = time.perf_counter()
    # in this dict we collect everything that we want to report.
    
    json_dict = {}
    json_dict['time_script_start'] = s
    json_dict['config'] = config

    if write_output_file:
        try:
            with open(config['out_file'], 'w+') as outf:
                pass
        except:
            raise ValueError("Output file path (-of/--out-file) improper or nonexistent.") 
        
    threshold = TRANSFORMATIONS[config['transformation']](config['threshold'])
    json_dict['config']['threshold_transformed'] = threshold


    ## Processing starts here:

    ## Load entities/samples as networkx graphs. labels contains label metadata.
    full_graph, part_graph, labels = make_graphs_from_sequences(config, threshold, json_dict, verbose)


    ## Let's look at the number of edges
    print("Full graph nr. of edges:", full_graph.number_of_edges())

    json_dict['graph_edges_start'] = full_graph.number_of_edges()
    json_dict['time_edges_complete'] = time.perf_counter()

    if config['save_checkpoint_path'] is not None:
        from .transformations import INVERSE_TRANSFORMATIONS
        from tqdm.auto import tqdm
        print(f'Saving edge list at {config["save_checkpoint_path"]} ...')
        with open(config['save_checkpoint_path'], 'w') as f:
            inv_tf = INVERSE_TRANSFORMATIONS[config['transformation']]
            for qry, lib, data in tqdm(full_graph.edges(data=True)):
                # we save the original metric. not the one that we transformed. So revert transformation.
                score = inv_tf(data['metric'])
                f.write(qry+ ',' + lib +',' + str(score) +'\n')


    
    ## Finally, let's partition this
    df = partition_and_remove(full_graph, part_graph, labels, json_dict, threshold, config, write_intermediate_file=False, verbose=verbose)

    ## clustering to outfile. This will probably change...
    if write_output_file:
        df.to_csv(config['out_file'])

    elapsed = time.perf_counter() - s
    json_dict['time_script_complete'] = time.perf_counter()

    if verbose:
        print(f"Graph-Part executed in {elapsed:0.2f} seconds.")

    if write_json_report:
        import json
        import os
        json.dump(json_dict, open(os.path.splitext(config['out_file'])[0]+'_report.json','w'))


    return df
