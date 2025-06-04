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

def partition_assignment_regression(cluster_vector, target_vector, n_partitions, n_bins=10, alpha=1.0, beta=1.0, n_mix=10):
    """
    Partition clustered data into N subsets while maintaining balance in both:
    - Regression target value distribution (via quantile-based binning)
    - Total sum of target values per partition

    Parameters:
    -----------
    cluster_vector : np.ndarray of shape (n_samples,)
        Array assigning each protein to a cluster (original from article Graph-Part).
        Note: Proteins in the same cluster are considered similar and must be kept in the same partition (if possible).

    target_vector : np.ndarray of shape (n_samples,)
        Continuous regression target values (e.g., termoestability).

    n_partitions : int
        Number of partitions (come from test , val ratio).

    n_bins : int (default=10)
        Number of bins to discretize the regression target values, using quantile binning.

    alpha : float (default=1.0)
        Weight for the divergence in bin distribution when calculating the partition score.

    beta : float (default=1.0)
        Weight for the deviation from balanced sum of target values when calculating the score.
    
    n_mix : int (default=10)
        Number of random swaps to perform to optimize the partition assignments.

    Returns:
    --------
    cl_number : np.ndarray of shape (n_samples,)
        An array indicating to which partition each protein is assigned .
    """
    
    # 1. Discretize the continuous target values into quantile-based bins
    #for now the strategy is quantile, but it can be changed to uniform or kmeans
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    label_bins = est.fit_transform(target_vector.reshape(-1, 1)).astype(int).flatten()
    with open("label_bins_output.txt", "w") as f:
        f.write("Label bins:\n")
        f.write(", ".join(map(str, label_bins)))  # Converte os valores para string e os separa por vírgula
    # 2. Get unique cluster IDs
    # this info is associated with the orignial code fro  graphpart article
    #where in each cluster all the proteins are similar
    u_cluster = np.unique(cluster_vector)

    # 3. Initialize tracking structures
    loc_bin_dist = np.zeros((n_partitions, n_bins))  # Bin distribution in each partition
    loc_sum = np.zeros(n_partitions)                # Sum of target values(continue values) in each partition
    cl_number = np.zeros(cluster_vector.shape[0])   # Output: partition ID for each sample

    # 4. Iterate through each cluster and assign to the best partition
    for i in u_cluster: #each cluster
        # Identify the positions (indices) of samples in this cluster
        positions = np.where(cluster_vector == i)[0] #analysis of each cluster from 0---end

        # Extract target values and corresponding bins for this cluster
        cluster_targets = target_vector[positions] #continue value for a protein that below to the cluster in analysis
        cluster_bins = label_bins[positions]#bins strategy
        cluster_sum = np.sum(cluster_targets)

        # Count the number of samples in each bin for this cluster, like 3 proteins in bin 0, 2 in bin 1 for the cluster A..
        bin_counts = np.zeros(n_bins)
        unique_bins, bin_freq = np.unique(cluster_bins, return_counts=True)
        bin_counts[unique_bins] = bin_freq

        # Find the best partition for this cluster
        best_score = float('inf')
        best_partition = None

        for p in range(n_partitions): # for each partition
            # a) Simulate bin distribution if we add this cluster to partition p
            bin_dist = loc_bin_dist[p] + bin_counts
            bin_dist_norm = bin_dist / np.sum(bin_dist)

            # b) Calculate global bin distribution if cluster is added to partition p
            overall_bin_dist = np.sum(loc_bin_dist, axis=0) + bin_counts
            overall_bin_dist /= np.sum(overall_bin_dist)

            # c) Compute L1 divergence between local and global bin distributions
            #we want to minimize the difference between the local and global bin distribution
            #avoiding the situation where all the proteins in a cluster are assigned to the same bin
            bin_div = np.sum(np.abs(bin_dist_norm - overall_bin_dist))

            # d) Compute difference in target sum from expected average
            temp_sum = loc_sum[p] + cluster_sum
            avg_sum = (np.sum(loc_sum) + cluster_sum) / n_partitions
            sum_diff = np.abs(temp_sum - avg_sum) #we want to minimize the difference between the local and global sum of target values

            # e) Final score combines bin divergence and target sum imbalance
            # The weights alpha and beta control the importance of each term
            score = alpha * bin_div + beta * sum_diff

            # f) Store the partition with the lowest score
            if score < best_score:
                best_score = score
                best_partition = p

        # 5. Assign the cluster to the chosen partition
        cl_number[positions] = best_partition #assign the cluster to the best partition
        loc_bin_dist[best_partition] += bin_counts #add the bin counts to the best partition
        loc_sum[best_partition] += cluster_sum #add the cluster sum to the best partition
    from collections import Counter

    # Contar a composição das partições
    partition_composition = Counter(cl_number)

    # Abrir um arquivo para salvar a saída
    with open("partition_composition.txt", "w") as f:
        f.write("Partitions created and their composition:\n")
        for partition, count in partition_composition.items():
            f.write(f"Partition {int(partition)}: {count} elements\n")

    cl_number, final_score = optimize_partitions(cl_number, cluster_vector, target_vector, label_bins, n_partitions, n_bins, alpha, beta, n_mix)
    return cl_number.astype(int)

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

def score_partitioning(df: pd.core.frame.DataFrame) -> float:    
    if 'sum' in df.columns:
        relevant_column = df['sum']
    else:
        raise ValueError("Expected 'sum' column in DataFrame for regression scoring.")
    
    s0 = relevant_column.shape[0]  # Número de linhas
    return float((relevant_column**(1/s0)).product())

def display_results_regression(
    part_graph: nx.classes.graph.Graph, 
    full_graph: nx.classes.graph.Graph,
    nr_of_parts: int,
    verbose: bool = True) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    """
    Display results for regression problems, focusing on continuous target values (label-val).

    Args:
        part_graph (nx.Graph): The partition graph with cluster assignments.
        full_graph (nx.Graph): The full graph with node attributes.
        nr_of_parts (int): Number of partitions.
        verbose (bool): If True, print the results.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame with node details and summary statistics.
    """
    # Create a DataFrame from the full graph nodes
    df = pd.DataFrame(((d) for n, d in full_graph.nodes(data=True)))
    df['cluster'] = [part_graph.nodes[n]['cluster'] for n in full_graph.nodes()]
    print(df['cluster'] )
    # Check if any partition is completely removed
    if len(df['cluster'].unique()) < nr_of_parts:
        error_string = f'''
        Impossible to generate the desired {nr_of_parts} partitions at the current partitioning threshold.
        Removal of sequences to achieve separation results in loss of {nr_of_parts - len(df['cluster'].unique())} complete partitions.
        '''
        raise RuntimeError(error_string)

    # Add node identifiers
    df['AC'] = [n for n in full_graph.nodes()]

    # Calculate statistics for each partition
    result = df.groupby('cluster')['label-val'].agg(['mean', 'sum', 'std', 'count']).T
    result['total'] = result.sum(axis=1)  # Add a total column for overall statistics

    if verbose:
        print(result)
        print()
        #print("Partitioning score:", score_partitioning(result.loc['sum']))
        print()

    return df, result   
    

def removal_needed(
    part_graph: nx.classes.graph.Graph, 
    full_graph: nx.classes.graph.Graph,
    threshold: float) -> bool:
    """ """
    return True


from .precomputed_utils import load_edge_list
from .mmseqs_utils import generate_edges_mmseqs
from .needle_utils import generate_edges_mp
from .needle_utils import generate_edges
from .mmseqs_needle_combined_utils import generate_edges_mmseqs_needle_combined

def make_graphs_from_sequences_regression(config: Dict[str, Any], threshold: float, json_dict: Dict[str, Any], verbose: bool = True) -> Tuple[nx.classes.graph.Graph, nx.classes.graph.Graph]:
    """
    This function performs the alignments and constructs the graphs for regression problems.

    Parameters:
    ------------
        config: dict
            A dictionary of all parameters needed to run graph-part.

        threshold: float
            The threshold to use for partitioning. Alignments 
            are discarded if their distance is above the threshold.

        json_dict: dict
            A dictionary to collect outputs for the final report.

        verbose: bool
            If True, print all processing steps to the command line.

    Returns:
    ------------
        full_graph: nx.classes.graph.Graph
            Networkx graph that has sequences as nodes and their distances as edge attributes.
        part_graph: nx.classes.graph.Graph
            Networkx graph that collects the final partition assignments.
    """
    # Load entities and initialize graphs
    full_graph, part_graph, _ = load_entities(config['fasta_file'], config['priority_name'], config['labels_name'])

    # Debugging: Print the range of continuous target values
    if verbose:
        label_vals = [full_graph.nodes[AC]['label-val'] for AC in full_graph.nodes()]
        print(f"Loaded {len(label_vals)} entities with label-val range: {min(label_vals)} to {max(label_vals)}")

    # Handle different alignment modes
    if config['alignment_mode'] == 'precomputed':
        print('Parsing edge list.')
        load_edge_list(config['edge_file'], full_graph, config['transformation'], threshold, config['metric_column'])
        elapsed_align = time.perf_counter() - json_dict['time_script_start']
        if verbose:
            print(f"Edge list parsing executed in {elapsed_align:0.2f} seconds.")

    elif config['alignment_mode'] == 'mmseqs2':
        generate_edges_mmseqs(config['fasta_file'], full_graph, config['transformation'], threshold, config['threshold'], denominator=config['denominator'], delimiter='|', is_nucleotide=config['nucleotide'], use_prefilter=config['prefilter'])
        elapsed_align = time.perf_counter() - json_dict['time_script_start']
        if verbose:
            print(f"Pairwise alignment executed in {elapsed_align:0.2f} seconds.")

    elif config['alignment_mode'] == 'needle' and config['threads'] > 1:
        print('Computing pairwise sequence identities.')
        generate_edges_mp(config['fasta_file'], full_graph, config['transformation'], threshold, denominator=config['denominator'], n_chunks=config['chunks'], n_procs=config['threads'], parallel_mode=config['parallel_mode'], triangular=config['triangular'], delimiter='|', 
                          is_nucleotide=config['nucleotide'], gapopen=config['gapopen'], gapextend=config['gapextend'], endweight=config['endweight'], endopen=config['endopen'], endextend=config['endextend'], matrix=config['matrix'])
        elapsed_align = time.perf_counter() - json_dict['time_script_start']
        if verbose:
            print(f"Pairwise alignment executed in {elapsed_align:0.2f} seconds.")

    elif config['alignment_mode'] == 'needle':
        print('Computing pairwise sequence identities.')
        generate_edges(config['fasta_file'], full_graph, config['transformation'], threshold, denominator=config['denominator'], delimiter='|',
                       is_nucleotide=config['nucleotide'], gapopen=config['gapopen'], gapextend=config['gapextend'], endweight=config['endweight'], endopen=config['endopen'], endextend=config['endextend'], matrix=config['matrix'])
        elapsed_align = time.perf_counter() - json_dict['time_script_start']
        if verbose:
            print(f"Pairwise alignment executed in {elapsed_align:0.2f} seconds.")

    elif config['alignment_mode'] == 'mmseqs2needle':
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
    else:
        raise NotImplementedError('Encountered unspecified alignment mode. This should never happen.')

    return full_graph, part_graph


def partition_and_remove_regression(full_graph: nx.classes.graph.Graph, part_graph: nx.classes.graph.Graph, json_dict: dict,
                         threshold: float, config: dict, write_intermediate_file: bool = False, verbose: bool = True) -> pd.core.frame.DataFrame:
    '''
    This function runs the core Graph-Part algorithm for regression problems. Its inputs are generated by
    `make_graphs_from_sequences` or another function that produces outputs of the same kind for non-sequence data.
    '''
    
    # Partition the data using regression-specific logic
    partition_data_regression(full_graph, part_graph, threshold, config['partitions'], config['initialization_mode'])
    
    # Display initial results for regression
    df, result = display_results_regression(part_graph, full_graph, config['partitions'], verbose=verbose)
    
    # Handle train, validation, and test splits if required
    if config['test_ratio'] > 0:
        train_val_test_split(part_graph, full_graph, threshold, config['test_ratio'], config['val_ratio'], config['partitions'])
        config['partitions'] = 3 if config['val_ratio'] > 0 else 2

    # Display results after train/val/test split
    df, result = display_results_regression(part_graph, full_graph, config['partitions'], verbose=verbose)
    
    # Optionally write intermediate results to a file
    if write_intermediate_file:
        df.to_csv(config['out_file'] + "pre-removal")
    print('Currently have this many samples:', full_graph.number_of_nodes())

    # Update JSON dictionary with pre-removal statistics
    json_dict['partitioning_pre_removal'] = result.to_json()
    json_dict['samples_pre_removal'] = full_graph.number_of_nodes()
    #json_dict['score_pre_removal'] = score_partitioning(result.loc['sum'])

    # Check if removal is needed
    if removal_needed(part_graph, full_graph, threshold):     
        print('Need to remove! Currently have this many samples:', full_graph.number_of_nodes())
        remover(full_graph, part_graph, threshold, json_dict, config['allow_moving'], True, verbose=verbose)    

    if removal_needed(part_graph, full_graph, threshold):   
        print('Need to remove priority! Currently have this many samples:', full_graph.number_of_nodes())
        remover(full_graph, part_graph, threshold, json_dict, config['allow_moving'], False, verbose=verbose)    

    print('After removal we have this many samples:', full_graph.number_of_nodes())

    # Display final results for regression
    df, result = display_results_regression(part_graph, full_graph, config['partitions'], verbose=verbose)

    # Update JSON dictionary with post-removal statistics
    json_dict['partitioning_after_removal'] = result.to_json()
    json_dict['samples_after_removal'] = full_graph.number_of_nodes()
    #json_dict['score_after_removal'] = score_partitioning(result.loc['sum'])

    # Final check for removal necessity
    if removal_needed(part_graph, full_graph, threshold):
        print("Something is wrong! Removal still needed!")
        json_dict['removal_needed_end'] = True
    else:
        json_dict['removal_needed_end'] = False

    return df

def run_partitioning_regression(config: Dict[str, Union[str, int, float, bool]], write_output_file: bool = True, write_json_report: bool = True, verbose: bool = True) -> pd.core.frame.DataFrame:
    '''
    Core Graph-Part partitioning function for regression problems. `config` contains all parameters passed from the command line
    or Python API. See `cli.py` for the definitions.  

    Parameters:
    -----------
    config:
        A dictionary of all parameters needed to run graph-part.
    write_output_file: bool
        If True, write final assignment table to disk.
    write_json_report: bool
        If True, write a report of all summary statistics. Used by the webserver.
    verbose: bool
        If True, print all processing steps to command line.
    '''

    s = time.perf_counter()
    # Initialize the JSON dictionary for reporting
    json_dict = {}
    json_dict['time_script_start'] = s
    json_dict['config'] = config

    # Validate output file path
    if write_output_file:
        try:
            with open(config['out_file'], 'w+') as outf:
                pass
        except:
            raise ValueError("Output file path (-of/--out-file) improper or nonexistent.") 

    # Apply the transformation to the threshold
    threshold = TRANSFORMATIONS[config['transformation']](config['threshold'])
    json_dict['config']['threshold_transformed'] = threshold

    ## Processing starts here:

    # Load entities/samples as networkx graphs
    full_graph, part_graph = make_graphs_from_sequences_regression(config, threshold, json_dict, verbose)

    # Debugging: Print the number of edges in the full graph
    print("Full graph nr. of edges:", full_graph.number_of_edges())
    json_dict['graph_edges_start'] = full_graph.number_of_edges()
    json_dict['time_edges_complete'] = time.perf_counter()

    # Optionally save the edge list to a checkpoint file
    if config['save_checkpoint_path'] is not None:
        from .transformations import INVERSE_TRANSFORMATIONS
        from tqdm.auto import tqdm
        print(f'Saving edge list at {config["save_checkpoint_path"]} ...')
        with open(config['save_checkpoint_path'], 'w') as f:
            inv_tf = INVERSE_TRANSFORMATIONS[config['transformation']]
            for qry, lib, data in tqdm(full_graph.edges(data=True)):
                # Save the original metric (reverting the transformation)
                score = inv_tf(data['metric'])
                f.write(qry + ',' + lib + ',' + str(score) + '\n')

    # Partition the graph and remove problematic nodes
    df = partition_and_remove_regression(full_graph, part_graph, json_dict, threshold, config, write_intermediate_file=False, verbose=verbose)

    # Save the final clustering to the output file
    if write_output_file:
        df.to_csv(config['out_file'])

    # Record the elapsed time
    elapsed = time.perf_counter() - s
    json_dict['time_script_complete'] = time.perf_counter()

    # Print execution time if verbose
    if verbose:
        print(f"Graph-Part executed in {elapsed:0.2f} seconds.")

    # Optionally write the JSON report
    if write_json_report:
        import json
        import os
        json.dump(json_dict, open(os.path.splitext(config['out_file'])[0] + '_report.json', 'w'))

    return df