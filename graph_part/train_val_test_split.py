'''
Code to generate train-val-test splits using Graph-Part.
We first partition the data into 10 or 20 partitions. These
are then combined to yield the splits. Then removal is applied,
as always.
'''
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from typing import List, Tuple


def check_train_val_test_args(args):
    '''If a train-val-test split is to be done, we
    need to fix the "partitions" to be compatible.'''
    if args.test_ratio*100 % 5 !=0 or args.val_ratio*100 % 5 != 0:
        raise NotImplementedError('Graph-Part currently only supports ratios that are a multiple of 0.05!')

    if args.test_ratio* 100 % 10 == 0 and args.val_ratio* 100 % 10 == 0:
        setattr(args, 'partitions', 10)
    else:
        setattr(args, 'partitions', 20)

    # if test_ratio is 0 but val_ratio is defined, just swap the two. Then everything works.
    if args.test_ratio ==0:
        setattr(args, 'test_ratio', args.val_ratio)
        setattr(args, 'val_ratio', 0.0)

def compute_partition_matrix_similar_sequences(full_graph: nx.classes.graph.Graph, part_graph: nx.classes.graph.Graph, n_partitions: int, threshold: float) -> np.ndarray:
    '''
    Compute a matrix of the partitions, evaluating which partitions contain more similar sequences.
    Goal: Ensure that when comparing the partitions, the similarity between sequences is lower (to avoid bias).
    Args:
        full_graph (nx.Graph): The full graph containing sequences and their connections.
        part_graph (nx.Graph): The partition graph containing the clusters of sequences, is like groups of similar sequences.
        n_partitions (int): Total number of partitions.
        threshold (float): Threshold to define connections based on 'metric' attribute.
    '''
    
    partition_connections_similar_sequences = np.zeros((n_partitions, n_partitions)) # matrix form nº partitions vs nº partitions
    cluster_sizes = Counter(part_graph.nodes[n]['cluster'] for n in full_graph.nodes())
    for n, d in full_graph.nodes(data=True): # n is the node, d is the data associated with the node
        # get the partition of the current(n) node 
        self_cluster = part_graph.nodes[n]['cluster'] 
        # get the neighbors of the current(n) node
        neighbours = nx.neighbors(full_graph, n)
        # full_graph[n][nb]['metric'] is the metric of the edge between the current node and the neighbor node
        # only nodes with connection with neighbors with metric < threshold are considered and counted; only are considered sequences that are different
        neighbour_clusters = Counter((part_graph.nodes[nb]['cluster'] for nb in neighbours if full_graph[n][nb]['metric'] < threshold))
        # so the real matrix is the number of connections between each partition
        for cl, count in neighbour_clusters.items(): # cl and count are the cluster/partition and the number of connections between the current node and the neighbor node
            partition_connections_similar_sequences[int(self_cluster), int(cl)] += count / (cluster_sizes[self_cluster] * cluster_sizes[cl])#normalized by the size of the clusters
    return partition_connections_similar_sequences # count of connections between partitions that are really similar in sequences

def compute_partition_matrix_different_sequences(full_graph: nx.classes.graph.Graph, part_graph: nx.classes.graph.Graph, n_partitions: int, threshold: float) -> np.ndarray:
    '''
    Compute a matrix of the partitions, evaluating which partitions contain more distinct sequences.
    Goal: Ensure that each set/group of partitionis includes sequences with some level of diversity.
    Args:
        full_graph (nx.Graph): The full graph containing sequences and their connections.
        part_graph (nx.Graph): The partition graph containing the clusters of sequences.
        n_partitions (int): Total number of partitions.
        threshold (float): Threshold to define connections based on 'metric' attribute.
    '''
    partition_connections_different_sequences = np.zeros((n_partitions, n_partitions)) 
    cluster_sizes = Counter(part_graph.nodes[n]['cluster'] for n in full_graph.nodes())
    for n, d in full_graph.nodes(data=True): 
        self_cluster = part_graph.nodes[n]['cluster'] 
        neighbours = nx.neighbors(full_graph, n)
        neighbour_clusters = Counter((part_graph.nodes[nb]['cluster'] for nb in neighbours if full_graph[n][nb]['metric'] >= threshold))
        for cl, count in neighbour_clusters.items(): 
            partition_connections_different_sequences[int(self_cluster), int(cl)] += count / (cluster_sizes[self_cluster] * cluster_sizes[cl])
    return partition_connections_different_sequences

def find_best_partition_combinations(partition_connections_similar_sequences: np.ndarray, partition_connections_different_sequences: np.ndarray, n_train: int, n_test: int, 
                                     n_val: int, label_counts: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    '''
    Brute force try all combinations of partitions to find the set of partitions that have:
      1- class balance (greater than 37 %)
      2- minimum connections between train, test, and val sets (to avoid bias)
      3- maximum diversity within each set (to ensure different sequences are included)
    This works because the expected number of partitions is low enough, e.g. steps of 10% or 5% -> max 20 partitions.
    Don't do this when using 1% steps, will explode.
    '''
    partitions = list(range(partition_connections_similar_sequences.shape[0]))
    total_labels = label_counts.sum(axis=0)
    total_labels = total_labels / total_labels.sum()

    def get_best_combination(partitions, n_train, n_test, n_val):
        best_combination = ([], [], [])  # To store the best combination of partitions for train, test, and val.
        best_score = -1  # Initialize with a score less than the minimum possible (e.g., 0).
        min_score_similarity_possible = float("inf")
        max_diversity_score = -float("inf")

        # List to store all combinations and their scores
        all_combinations = []

        for train_comb in combinations(partitions, n_train):#all the possible combinations
            remaining_after_train = [p for p in partitions if p not in train_comb]
            for test_comb in combinations(remaining_after_train, n_test):
                val_comb = [p for p in remaining_after_train if p not in test_comb] 

                max_paritions_metric_below_tresh = partition_connections_similar_sequences.sum() #below treshold, similar sequences are considered (similar)
                max_partitions_metric_supeior_tresh = partition_connections_different_sequences.sum() #higher than treshold, different sequences are considered (diversity)

                # Calculate the class balance  for the combination.
                train_distribution = label_counts[list(train_comb)].sum(axis=0)
                test_distribution = label_counts[list(test_comb)].sum(axis=0)
                val_distribution = label_counts[list(val_comb)].sum(axis=0)

                train_distribution = train_distribution / train_distribution.sum()
                test_distribution = test_distribution / test_distribution.sum()
                val_distribution = val_distribution / val_distribution.sum()
                
                # Ensure no train/test/val set has a class imbalance greater than 35% 
                #need do ajust in the future
                if np.any(train_distribution < 0.37) or np.any(test_distribution < 0.37) or np.any(val_distribution < 0.37):
                    continue
                
                # Penalization for connections between sets, ww want to minimize this because we want to avoid bias
                train_test_penalty = partition_connections_similar_sequences[train_comb, :][:, test_comb].sum()
                train_val_penalty = partition_connections_similar_sequences[train_comb, :][:, val_comb].sum()
                test_val_penalty = partition_connections_similar_sequences[test_comb, :][:, val_comb].sum()
                penalty = (train_test_penalty + train_val_penalty + test_val_penalty) / max_paritions_metric_below_tresh

                 # Calculate the diversity score for each set combination, we want maximize this
                train_diversity = partition_connections_different_sequences[train_comb, :][:, train_comb].sum()
                test_diversity = partition_connections_different_sequences[test_comb, :][:, test_comb].sum()
                val_diversity = partition_connections_different_sequences[val_comb, :][:, val_comb].sum()
                diversity_score = (train_diversity + test_diversity + val_diversity) / max_partitions_metric_supeior_tresh

                #the rule is to minimize the penalty and maximize the diversity score, in this order
                if penalty < min_score_similarity_possible:
                    min_score_similarity_possible = penalty
                    best_combination = (train_comb, test_comb, val_comb)
                elif penalty == min_score_similarity_possible:
                    if diversity_score > max_diversity_score:
                        max_diversity_score = diversity_score
                        best_combination = (train_comb, test_comb, val_comb)
                
                # Store the combination and its scores
                all_combinations.append({
                    'train_comb': train_comb,
                    'test_comb': test_comb,
                    'val_comb': val_comb,
                    'diversity_score': diversity_score,
                    'external_penalty': penalty
                })

        return best_combination

    # Find the best combination of partitions for train, test, and validation sets.
    train_partitions, test_partitions, val_partitions = get_best_combination(partitions, n_train, n_test, n_val)

    return train_partitions, test_partitions, val_partitions


def train_val_test_split(part_graph: nx.classes.graph.Graph, 
                         full_graph: nx.classes.graph.Graph, 
                         threshold: float, 
                         test_ratio: float,
                         val_ratio: float, 
                         n_partitions: int = 10) -> None:
    """
    Splits the graph into train, test, and validation sets, ensuring balance in class distribution and minimizing bias.
    
    Args:
        part_graph (nx.Graph): The partition graph containing the clusters of sequences.
        full_graph (nx.Graph): The full graph containing sequences and their connections.
        threshold (float): Threshold to define connections based on 'metric' attribute.
        test_ratio (float): Ratio of partitions for the test set.
        val_ratio (float): Ratio of partitions for the validation set.
        n_partitions (int): Total number of partitions.
    """
    # Step 1: Compute the number of partitions for each set.
    n_train = int(round(n_partitions * (1 - val_ratio - test_ratio)))  
    n_test = int(n_partitions * test_ratio)
    n_val = int(n_partitions * val_ratio)
    # Step 2: Compute the similarity matrix between partitions.
    partition_connections = compute_partition_matrix_similar_sequences(full_graph, part_graph, n_partitions, threshold)
    partition_connections_different_sequences = compute_partition_matrix_different_sequences(full_graph, part_graph, n_partitions, threshold)
    
    
    # Step 3: Compute the label distribution (label_counts) for each partition.
    # Create a DataFrame with cluster and label information for all nodes.
    df = pd.DataFrame(((d) for n, d in full_graph.nodes(data=True)))
    df['cluster'] = [part_graph.nodes[n]['cluster'] for n in full_graph.nodes()]
    df['AC'] = [n for n in full_graph.nodes()]

    # Group by cluster and label value to count occurrences of each label in each partition.
    df_label_counts = df.groupby(['cluster', 'label-val'])['AC'].count().reset_index()
    label_counts = df_label_counts.pivot_table(values='AC', columns=['label-val'], index=['cluster'], fill_value=0).to_numpy() #matrix with the number of each label in each partition
    
    # Step 4: Find the best combinations for train, test, and validation sets.
    train_partitions, test_partitions, val_partitions = find_best_partition_combinations(
        partition_connections, partition_connections_different_sequences, n_train, n_test, n_val, label_counts
    )

    # Step 5: Compute new statistics for each set (train, test, val) based on the chosen partitions.
    train_statistics = df.loc[df['cluster'].isin(train_partitions)].sum(axis=0)
    train_attributes = {'cluster': 0.0, 'C-size': train_statistics.get('AC', 0), 'label-counts': train_statistics.get('label-val', np.array([]))}
    test_statistics = df.loc[df['cluster'].isin(test_partitions)].sum(axis=0)
    test_attributes = {'cluster': 1.0, 'C-size': test_statistics.get('AC', 0), 'label-counts': test_statistics.get('label-val', np.array([]))}
    val_statistics = df.loc[df['cluster'].isin(val_partitions)].sum(axis=0)
    val_attributes = {'cluster': 2.0, 'C-size': val_statistics.get('AC', 0), 'label-counts': val_statistics.get('label-val', np.array([]))}

    # Step 6: Update the part_graph with the new train, test, and validation assignments.
    for n, d in part_graph.nodes(data=True):
        if int(d['cluster']) in train_partitions:
            nx.set_node_attributes(part_graph, {n: train_attributes})
        elif int(d['cluster']) in test_partitions:
            nx.set_node_attributes(part_graph, {n: test_attributes})
        else:
            nx.set_node_attributes(part_graph, {n: val_attributes})

# Not used.
def find_best_partition_combinations_heuristic(partition_connections: np.ndarray, n_train: int, n_test: int):
    '''
    Brute force try combinations of partitions to find the set of partitions that have the maximum connections to each other.
    This here still tries a lot, but is not exhaustive. 
    Version above: O(min(n^k, n^(n-k)))
    This:        : O(n*n*k)
    As long as we have 10 or 20 partitions, n choose k is fine. This works too, but untested for larger n.
    '''
    # 1. Get train partitions
    best_combination = []
    best_score = 0
    current_score = 0
    
    # Try each partition as the starting point. complexity n*n*k
    for partition_id in range(partition_connections.shape[0]): # complexity n
        current_combination = []
        current_combination.append(partition_id)
        
        # Add partitions to the set until the size is reached.
        while(len(current_combination)<n_train): #complexity k
            
            # find the best partition to add.
            best_connections = 0
            best_connected = None
            for i in range(partition_connections.shape[0]): # complexity n
                #print('Current', current_combination, 'testing', i, 'best', best_connections)
                if i in current_combination:
                   # print('i in current')
                    continue
                
                connections = partition_connections[i, current_combination].sum()
                #print(f'Connections for {i}: {connections}')
                if connections > best_connections:
                    best_connected = i
                    best_connections = connections
            
            current_combination.append(best_connected)
        
        # Got a set. Keep if best, else discard.
        # We double count here (do np.triu to avoid), but it does not matter if we do it for all.
        current_score = partition_connections[current_combination,:][:,current_combination].sum()
        if current_score>best_score:
            best_combination = current_combination
            best_score = current_score
        
    return best_combination