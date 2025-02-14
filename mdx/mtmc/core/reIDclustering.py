import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

class ReIDClusterer:
    """
    Module for clustering behaviors based on appearance embeddings and temporal information.

    :param dict config: configuration for the app
    ::

        clusterer = ReIDClusterer(config)
    """

    def __init__(self, config) -> None:
        self.config = config

    def _calculate_euclidean_dist(self, embedding_a: np.array, embedding_b: np.array) -> float:
        """
        Calculates normalized Euclidean distance between two embeddings.

        :param np.array embedding_a: the first embedding vector
        :param np.array embedding_b: the second embedding vector
        :return: Euclidean distance
        :rtype: float
        """
        return np.clip((2 - (2 * np.dot(embedding_a, embedding_b.T))), 0, 4) / 4.

    def _cluster_embeddings(self, behavior_keys: List[str], embedding_array: np.array, num_clusters: Optional[int]) -> Dict[int, List[str]]:
        """
        Clusters embeddings using an appearance-based metric.

        :param List[str] behavior_keys: list of behavior keys
        :param np.array embedding_array: array of embeddings
        :param Optional[int] num_clusters: number of clusters, or None
        :return: map from global IDs to clusters
        :rtype: Dict[int, List[str]]
        """
        logging.info("Clustering embeddings based on appearance...")

        if self.config.clustering.clusteringAlgo == "AgglomerativeClustering":
            clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(embedding_array)
        elif self.config.clustering.clusteringAlgo == "HDBSCAN":
            min_cluster_size = self.config.clustering.hdbscanMinClusterSize
            clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(embedding_array)
        else:
            logging.error(f"ERROR: Unknown clustering algorithm {self.config.clustering.clusteringAlgo}.")
            exit(1)

        cluster_labels = clustering.labels_
        map_global_id_to_cluster: Dict[int, List[str]] = defaultdict(list)
        for i in range(len(behavior_keys)):
            if cluster_labels[i] >= 0:  # Ignore noise points
                map_global_id_to_cluster[cluster_labels[i]].append(behavior_keys[i])

        logging.info(f"No. of clusters: {max(cluster_labels) + 1}")
        return map_global_id_to_cluster

    def _find_coexisting_behaviors(self, map_behavior_key_to_behavior: Dict[str, Dict], behavior_keys: List[str]) -> List[List[str]]:
        """
        Finds behaviors co-existing at the same time within the same sensor.

        :param Dict[str, Dict] map_behavior_key_to_behavior: map from behavior keys to behavior information
        :param List[str] behavior_keys: list of behavior keys
        :return: co-existing behavior groups
        :rtype: List[List[str]]
        """
        logging.info("Finding co-existing behaviors...")

        map_sensor_id_to_behavior_keys: Dict[str, List[str]] = defaultdict(list)
        for behavior_key in behavior_keys:
            behavior = map_behavior_key_to_behavior[behavior_key]
            map_sensor_id_to_behavior_keys[behavior['sensorId']].append(behavior_key)

        coexisting_behavior_groups = list(map_sensor_id_to_behavior_keys.values())
        logging.info(f"No. of co-existing behavior groups: {len(coexisting_behavior_groups)}")

        return coexisting_behavior_groups

    def _reassign_clusters(self, behavior_keys: List[str], embedding_array: np.array, cluster_means: Dict[int, np.array]) -> Dict[int, List[str]]:
        """
        Reassigns embeddings to clusters using the Hungarian algorithm based on appearance distance.

        :param List[str] behavior_keys: list of behavior keys
        :param np.array embedding_array: array of embeddings
        :param Dict[int, np.array] cluster_means: map of cluster IDs to mean embeddings
        :return: updated map from cluster IDs to behavior keys
        :rtype: Dict[int, List[str]]
        """
        logging.info("Reassigning embeddings to clusters...")

        # Calculate appearance distances
        dists = {}
        for i, embedding in enumerate(embedding_array):
            for cluster_id, cluster_mean in cluster_means.items():
                dists[(i, cluster_id)] = self._calculate_euclidean_dist(embedding, cluster_mean)

        # Build cost matrix
        num_embeddings = embedding_array.shape[0]
        num_clusters = len(cluster_means)
        cost_matrix = np.full((num_embeddings, num_clusters), np.inf)
        for (i, cluster_id), dist in dists.items():
            cost_matrix[i, cluster_id] = dist

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Reassign embeddings to clusters
        updated_clusters: Dict[int, List[str]] = defaultdict(list)
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.config.clustering.reassignmentDistThresh:
                updated_clusters[col].append(behavior_keys[row])

        return updated_clusters

    def cluster(self, behaviors: List[Dict]) -> Dict[int, List[str]]:
        """
        Main method to cluster embeddings based on appearance and temporal information.

        :param List[Dict] behaviors: list of behaviors, each containing sensorId, objectId, timestamp, and embeddings
        :return: map from cluster IDs to behavior keys
        :rtype: Dict[int, List[str]]
        """
        logging.info("Starting clustering process...")

        # Map behavior keys to behaviors
        map_behavior_key_to_behavior: Dict[str, Dict] = {}
        behavior_keys = []
        embeddings = []

        for behavior in behaviors:
            behavior_key = f"{behavior['sensorId']}-{behavior['objectId']}-{behavior['timestamp']}"
            map_behavior_key_to_behavior[behavior_key] = behavior
            behavior_keys.append(behavior_key)
            embeddings.append(behavior['embedding'])

        embedding_array = np.array(embeddings)

        # Initial clustering
        num_clusters = self.config.clustering.numClusters
        clusters = self._cluster_embeddings(behavior_keys, embedding_array, num_clusters)

        # Compute initial cluster means
        cluster_means: Dict[int, np.array] = {}
        for cluster_id, keys in clusters.items():
            cluster_indices = [behavior_keys.index(key) for key in keys]
            cluster_means[cluster_id] = np.mean(embedding_array[cluster_indices], axis=0)

        # Reassign clusters based on appearance
        if self.config.clustering.reassignmentEnabled:
            clusters = self._reassign_clusters(behavior_keys, embedding_array, cluster_means)

        return clusters
