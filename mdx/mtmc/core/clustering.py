# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
import bisect
import math
# import multiprocessing as mp
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
# from sklearn.cluster import Birch, OPTICS, DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
from ortools.linear_solver import pywraplp

from mdx.mtmc.config import AppConfig
from mdx.mtmc.schema import Behavior, MTMCObject, MTMCStateObjectPlus
from mdx.mtmc.core.data import normalize_vector


class Clusterer:
    """
    Module to cluster behaviors

    :param dict config: configuration for the app
    ::

        clusterer = Clusterer(config)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        if not 0 <= self.config.clustering.spatioTemporalDistLambda <= 1:
            logging.error(f"ERROR: The spatioTemporalDistLambda {self.config.clustering.spatioTemporalDistLambda} must be within [0, 1].")
            exit(1)

    def _sort_behavior_keys(self, behavior_keys: List[str]) -> List[str]:
        """
        Sorts behavior keys

        :param List[str] behavior_keys: behavior keys (sensor-object-timestamp IDs)
        :return: sorted behavior keys
        :rtype: List[str]
        """
        map_behavior_tokens_to_key: Dict[Tuple[str, int, datetime], str] = dict()

        for behavior_key in behavior_keys:
            behavior_key_tokens = behavior_key.split(" #-# ")
            behavior_tokens = (datetime.fromisoformat(behavior_key_tokens[2][:-1]), behavior_key_tokens[0], int(behavior_key_tokens[1]))
            map_behavior_tokens_to_key[behavior_tokens] = behavior_key

        return [map_behavior_tokens_to_key[behavior_tokens] for behavior_tokens in sorted(map_behavior_tokens_to_key.keys())]

    def _sample_overlapping_locations(self, behavior: Behavior, overlap_timestamp: datetime, overlap_end: datetime) -> List[List[float]]:
        """
        Samples locations of a behavior at the overlapping time span

        :param Behavior behavior: behavior object
        :param datetime overlap_timestamp: starting timestamp of the overlap
        :param datetime overlap_end: ending timestamp of the overlap
        :return: sampled list of locations and their timestamps
        :rtype: Tuple[List[List[float]],List[datetime]]
        """
        timestamps = [ts for i, ts in enumerate(behavior.timestamps) if behavior.locationMask[i]]
        idx_start = bisect.bisect_left(timestamps, overlap_timestamp)
        idx_end = bisect.bisect_left(timestamps, overlap_end)
        del timestamps
        return behavior.locations[idx_start: idx_end + 1]

    def _interpolate_and_average_locations(self, aggregated_locations: List[List[List[float]]]) -> List[List[float]]:
        """
        Interpolates and computes the mean of the aggregated locations

        :param List[List[List[float]]] aggregated_locations: aggregated locations that may have different lengths
        :return: averaged locations
        :rtype: List[List[float]]
        """
        if len(aggregated_locations) == 0:
            return list()

        # Find the maximum length among all aggregated locations
        max_length = max(len(locations) for locations in aggregated_locations)

        # Interpolate each list of locations to have the same (max) length
        interpolated_aggregated_locations: List[List[zip]] = list()
        for locations in aggregated_locations:
            # Separate x and y coordinates
            x_coords, y_coords = zip(*locations)

            # Generate old and new indices for interpolation
            old_indices = np.linspace(0, 1, len(locations))
            new_indices = np.linspace(0, 1, max_length)

            # Interpolate x and y separately
            interpolated_x_coords = np.interp(new_indices, old_indices, x_coords)
            interpolated_y_coords = np.interp(new_indices, old_indices, y_coords)

            # Combine x and y into locations
            interpolated_locations = list(zip(interpolated_x_coords, interpolated_y_coords))
            interpolated_aggregated_locations.append(interpolated_locations)

        # Return the mean locations
        return np.mean(interpolated_aggregated_locations, axis=0).tolist()

    def _calculate_spatio_temporal_dist_between_trajectories(self, locations_a: List[List[float]], locations_b: List[List[float]]) -> Optional[float]:
        """
        Calculates spatio-temporal distance between two lists of locations

        :param List[List[float]] locations_a: the first list of locations
        :param List[List[float]] locations_b: the second list of locations
        :return: spatio-temporal distance
        :rtype: Optional[float]
        """
        if (len(locations_a) == 0) or (len(locations_b) == 0):
            return None

        # Find the maximum length between the two lists of locations
        length_a = len(locations_a)
        length_b = len(locations_b)
        is_swapped = False
        if length_a >= length_b:
            max_length = length_a
            min_length = length_b
            long_locations = locations_a
            short_locations = locations_b
        else:
            max_length = length_b
            min_length = length_a
            long_locations = locations_b
            short_locations = locations_a
            is_swapped = True

        # Interpolate the shorter list of locations
        x_coords, y_coords = zip(*short_locations)
        old_indices = np.linspace(0, 1, min_length)
        new_indices = np.linspace(0, 1, max_length)
        interpolated_x_coords = np.interp(new_indices, old_indices, x_coords)
        interpolated_y_coords = np.interp(new_indices, old_indices, y_coords)
        interpolated_locations = list(zip(interpolated_x_coords, interpolated_y_coords))

        # Return directed Hausdorff distance
        if self.config.clustering.spatioTemporalDistType == "Hausdorff":
            if not is_swapped:
                return directed_hausdorff(np.array(long_locations), np.array(interpolated_locations))[0]
            else:
                return directed_hausdorff(np.array(interpolated_locations), np.array(long_locations))[0]

        # Return the mean of (squared) Euclidean distance
        elif self.config.clustering.spatioTemporalDistType == "pairwise":
            return np.mean(np.sqrt(np.sum((np.array(long_locations) - np.array(interpolated_locations)) ** 2, axis=1)))

        else:
            logging.error(f"ERROR: Unkown spatio-temporal distance type {self.config.clustering.spatioTemporalDistType}.")
            exit(1)

    def _calculate_spatio_temporal_dist_between_points(self, loc_dir_vec_a: List[float], loc_dir_vec_b: List[float]) -> float:
        """
        Calculates spatio-temporal distance between two points represented by location-direction vectors

        :param List[float] loc_dir_vec_a: the first location-direction vector
        :param List[float] loc_dir_vec_b: the second location-direction vector
        :return: spatio-temporal distance
        :rtype: float
        """
        # Extract locations
        x_a, y_a = loc_dir_vec_a[:2]
        x_b, y_b = loc_dir_vec_b[:2]

        # Calculate Euclidean distance
        spatial_dist = np.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

        # Initialize direction influence to 0
        direction_influence = 0

        # Check if direction vectors are available and not too small
        if (len(loc_dir_vec_a) == 4) and (len(loc_dir_vec_b) == 4):
            delta_x_a, delta_y_a = loc_dir_vec_a[2:]
            delta_x_b, delta_y_b = loc_dir_vec_b[2:]
            magnitude_a = np.sqrt((delta_x_a ** 2) + (delta_y_a ** 2))
            magnitude_b = np.sqrt((delta_x_b ** 2) + (delta_y_b ** 2))

            # Only consider direction if both magnitudes are significant
            spatio_temporal_dir_magnitude_thresh = self.config.clustering.spatioTemporalDirMagnitudeThresh
            if (magnitude_a > spatio_temporal_dir_magnitude_thresh) and \
                (magnitude_b > spatio_temporal_dir_magnitude_thresh):
                # Normalize direction vectors
                norm_dir_a = (delta_x_a / magnitude_a, delta_y_a / magnitude_a)
                norm_dir_b = (delta_x_b / magnitude_b, delta_y_b / magnitude_b)

                # Calculate angle difference between directions
                dot_product = np.dot(norm_dir_a, norm_dir_b)
                angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip for domain errors

                # Influence of direction on the final distance
                direction_influence = angle_diff / np.pi  # Normalize angle difference to [0, 1]

        # Combine spatial distance and direction influence
        spatio_temporal_dist = spatial_dist * (1. + direction_influence)

        return spatio_temporal_dist

    def _calculate_euclidean_dist(self, embedding_a: np.array, embedding_b: np.array) -> float:
        """
        Calculates (squared) Euclidean distance between two normalized embedding vectors

        :param np.array embedding_a: the first embedding vector
        :param np.array embedding_b: the second embedding vector
        :return: Euclidean distance
        :rtype: float
        """
        return np.clip((2 - (2 * np.dot(embedding_a, embedding_b.T))), 0, 4) / 4.

    def _count_clusters(self, embedding_array: np.array) -> int:
        """
        Counts clusters for agglomerative clustering

        :param np.array embedding_array: list of embeddings
        :return: count of MTMC objects
        :rtype: int
        """
        logging.info(f"Counting clusters for agglomerative clustering...")

        # for dist_thresh in np.arange(0.1, 5.1, 0.1):
        #     clustering = AgglomerativeClustering(distance_threshold=dist_thresh,
        #                                          n_clusters=None).fit(embedding_array)
        #     num_clusters = max(clustering.labels_) + 1
        #     logging.info(f"No. clusters: {num_clusters} @ distance threshold: {dist_thresh}")
        # exit(0)

        clustering = AgglomerativeClustering(distance_threshold=self.config.clustering.agglomerativeClusteringDistThresh,
                                             n_clusters=None).fit(embedding_array)
        num_clusters = max(clustering.labels_) + 1
        del clustering
        logging.info(f"No. clusters by counting: {num_clusters}")

        return num_clusters

    def _cluster_embeddings(self, behavior_keys: List[str], embedding_array: np.array, num_clusters: Optional[int]) -> Dict[int, List[str]]:
        """
        Clusters embeddings

        :param List[str] behavior_keys: list of behavior keys (sensor-object-timestamp IDs)
        :param np.array embedding_array: list of embeddings
        :param Optional[int] num_clusters: number of clusters or None
        :return: map from global IDs to clusters
        :rtype: Dict[int,List[str]
        """
        logging.info(f"Clustering embeddings...")

        if self.config.clustering.clusteringAlgo == "AgglomerativeClustering":
            clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(embedding_array)

        elif self.config.clustering.clusteringAlgo == "HDBSCAN":
            # for min_cluster_size in range(2, 11):
            #     clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(embedding_array)
            #     num_clusters = max(clustering.labels_) + 1
            #     logging.info(f"No. clusters: {num_clusters} @ min cluster size: {min_cluster_size}")
            # exit(0)

            min_cluster_size = self.config.clustering.hdbscanMinClusterSize
            if min_cluster_size > embedding_array.shape[0]:
                min_cluster_size = embedding_array.shape[0]
            clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(embedding_array)

        else:
            logging.error(f"ERROR: Unkown clustering algorithm {self.config.clustering.clusteringAlgo}.")
            exit(1)

        # clustering = Birch(n_clusters=num_clusters).fit(embedding_array)
        # clustering = OPTICS().fit(embedding_array)
        # clustering = DBSCAN().fit(embedding_array)
        # clustering = hdbscan.HDBSCAN().fit_predict(embedding_array)

        cluster_labels = clustering.labels_
        map_global_id_to_cluster: Dict[int, List[str]] = defaultdict(list)
        for i in range(len(behavior_keys)):
            if cluster_labels[i] >= 0:
                map_global_id_to_cluster[cluster_labels[i]].append(behavior_keys[i])
        logging.info(f"No. clusters: {max(cluster_labels) + 1}")
        del clustering
        del cluster_labels

        return map_global_id_to_cluster

    def _compute_mean_embeddings(self, behavior_keys: List[str], embedding_array: np.array, map_global_id_to_cluster: Dict[int, List[str]]) -> Dict[int, np.array]:
        """
        Groups embeddings and compute their mean embedding for each cluster

        :param List[str] behavior_keys: list of behavior keys (sensor-object-timestamp IDs)
        :param np.array embedding_array: list of embeddings
        :param Dict[int,List[str]] map_global_id_to_cluster: map from global IDs to clusters
        :return: map from global IDs to mean embeddings
        :rtype: Dict[int,np.array]
        """
        logging.info(f"Grouping clusters...")

        # Map behavior keys to global IDs of clusters
        map_behavior_key_to_global_ids: Dict[str, List[int]] = defaultdict(list)
        for global_id in map_global_id_to_cluster.keys():
            for behavior_key in map_global_id_to_cluster[global_id]:
                map_behavior_key_to_global_ids[behavior_key].append(global_id)

        # Group embeddings for each cluster
        map_global_id_to_mean_embedding: Dict[int, List[np.array]] = defaultdict(list)
        for i in range(len(behavior_keys)):
            for global_id in map_behavior_key_to_global_ids[behavior_keys[i]]:
                map_global_id_to_mean_embedding[global_id].append(embedding_array[i])
        del map_behavior_key_to_global_ids

        # Compute mean embedding for each cluster
        global_ids_to_be_removed: List[int] = list()
        for global_id in map_global_id_to_mean_embedding.keys():
            if len(map_global_id_to_mean_embedding[global_id]) > 0:
                map_global_id_to_mean_embedding[global_id] = np.sum(map_global_id_to_mean_embedding[global_id], axis=0)
                map_global_id_to_mean_embedding[global_id] = normalize_vector(map_global_id_to_mean_embedding[global_id])
            else:
                global_ids_to_be_removed.append(global_id)

        # Remove empty mean embeddings
        for global_id in global_ids_to_be_removed:
            map_global_id_to_mean_embedding.pop(global_id, None)
        del global_ids_to_be_removed

        return map_global_id_to_mean_embedding

    def _find_coexisting_behaviors(self, map_behavior_key_to_behavior: Dict[str, Behavior], behavior_keys: List[str]) -> List[List[str]]:
        """
        Finds behaviors co-existing at the same time within the same sensor

        :param Dict[str,Behavior] map_behavior_key_to_behavior: map from behavior keys (sensor-object-timestamp IDs) to behavior objects
        :param List[str] behavior_keys: list of behavior keys (sensor-object-timestamp IDs)
        :return: co-existing behavior groups
        :rtype: List[List[str]]
        """
        logging.info(f"Finding co-existing behaviors...")

        if len(behavior_keys) == 1:
            logging.info("No. co-existing behavior groups: 1")
            return [[behavior_keys[0]]]

        coexisting_behavior_group_list: List[List[str]] = list()

        # Group timestamps and behavior keys by sensors
        map_sensor_id_to_timestamps: Dict[str, Set[datetime]] = defaultdict(set)
        map_sensor_id_to_behavior_keys: Dict[str, List[str]] = defaultdict(list)
        for behavior_key in behavior_keys:
            map_sensor_id_to_timestamps[map_behavior_key_to_behavior[behavior_key].sensorId].add(map_behavior_key_to_behavior[behavior_key].timestamp)
            map_sensor_id_to_timestamps[map_behavior_key_to_behavior[behavior_key].sensorId].add(map_behavior_key_to_behavior[behavior_key].end)
            map_sensor_id_to_behavior_keys[map_behavior_key_to_behavior[behavior_key].sensorId].append(behavior_key)

        # Find co-existing groups
        for sensor_id in map_sensor_id_to_timestamps.keys():
            coexisting_behavior_groups: Set[Tuple[str]] = set()
            for timestamp in map_sensor_id_to_timestamps[sensor_id]:
                coexisting_behavior_group: List[str] = list()
                for behavior_key in map_sensor_id_to_behavior_keys[sensor_id]:
                    if map_behavior_key_to_behavior[behavior_key].timestamp <= timestamp <= map_behavior_key_to_behavior[behavior_key].end:
                        coexisting_behavior_group.append(behavior_key)
                coexisting_behavior_groups.add(tuple(self._sort_behavior_keys(coexisting_behavior_group)))
                del coexisting_behavior_group

            # Reduce co-existing behavior groups
            if len(coexisting_behavior_groups) > 1:
                coexisting_behavior_groups = [frozenset(coexisting_behavior_group) for coexisting_behavior_group in coexisting_behavior_groups]
                coexisting_behavior_groups_to_remove = set()
                for i in range(0, len(coexisting_behavior_groups) - 1):
                    for j in range(i + 1, len(coexisting_behavior_groups)):
                        if coexisting_behavior_groups[i].issubset(coexisting_behavior_groups[j]):
                            coexisting_behavior_groups_to_remove.add(coexisting_behavior_groups[i])
                        elif coexisting_behavior_groups[j].issubset(coexisting_behavior_groups[i]):
                            coexisting_behavior_groups_to_remove.add(coexisting_behavior_groups[j])
                coexisting_behavior_groups = set(coexisting_behavior_groups) - coexisting_behavior_groups_to_remove
                del coexisting_behavior_groups_to_remove

            coexisting_behavior_group_list.extend([list(coexisting_behavior_group) for coexisting_behavior_group in list(coexisting_behavior_groups)])

        del map_sensor_id_to_timestamps
        del map_sensor_id_to_behavior_keys

        logging.info(f"No. co-existing behavior groups: {len(coexisting_behavior_group_list)}")

        return coexisting_behavior_group_list

    def _calculate_appearance_dists(self, map_behavior_key_to_behavior: Dict[str, Behavior],
                                    map_global_id_to_mean_embedding: Dict[int, np.array],
                                    matched_behavior_key_set: Set[str]) -> Dict[Tuple[str, int], float]:
        """
        Computes appearance distances between behaviors and the mean embeddings of clusters

        :param Dict[str,Behavior] map_behavior_key_to_behavior: map from behavior keys (sensor-object-timestamp IDs) to behavior objects
        :param Dict[int,np.array] map_global_id_to_mean_embedding: map from global IDs to mean embeddings
        :param Set[str] matched_behavior_key_set: set of matched behaviors
        :return: map from pairs of behavior keys and global IDs to appearance distance
        :rtype: Dict[Tuple[str,int],float]
        """
        logging.info(f"Computing appearance distances...")
        map_key_id_pair_to_appearance_dist: Dict[Tuple[str, int], float] = dict()
        for behavior_key in map_behavior_key_to_behavior.keys():
            if behavior_key in matched_behavior_key_set:
                continue
            for global_id in map_global_id_to_mean_embedding.keys():
                map_key_id_pair_to_appearance_dist[(behavior_key, global_id)] = \
                    self._calculate_euclidean_dist(np.array(map_behavior_key_to_behavior[behavior_key].embeddings[0]), map_global_id_to_mean_embedding[global_id])
        return map_key_id_pair_to_appearance_dist

    def _reassign_coexisting_behavior_group(self, coexisting_behavior_group: List[str],
                                            map_behavior_key_to_behavior: Dict[str, Behavior],
                                            sorted_global_ids: List[int],
                                            map_global_id_to_cluster: Dict[int, List[str]],
                                            map_key_id_pair_to_appearance_dist: Dict[Tuple[str, int], float]) -> \
                                                Dict[str, List[int]]:
        """
        Re-assigns a co-existing behavior group based on the Hungarian algorithm

        :param List[str] coexisting_behavior_group: co-existing group of behavior keys
        :param Dict[str,Behavior] map_behavior_key_to_behavior: map from behavior keys (sensor-object-timestamp IDs) to behavior objects
        :param List[int] sorted_global_ids: sorted global IDs
        :param Dict[int,List[str]] map_global_id_to_cluster: map from global IDs to clusters
        :param Dict[Tuple[str,int],float] map_key_id_pair_to_appearance_dist: map from pairs of behavior keys and global IDs to appearance distance
        :return: map from behavior keys to global IDs
        :rtype: Dict[int,List[str]]
        """
        map_behavior_key_to_global_ids: Dict[str, List[int]] = defaultdict(list)

        # Initialize the map from pairs of behavior keys and global IDs to distance
        map_key_id_pair_to_dist: Dict[Tuple[str, int], Optional[float]] = dict()
        for behavior_key in coexisting_behavior_group:
            for global_id in sorted_global_ids:
                map_key_id_pair_to_dist[behavior_key, global_id] = None

        # Calculate spatio-temporal distance
        max_spatio_temporal_dist = None
        if self.config.clustering.spatioTemporalDistLambda > 0:
            sensor_id = map_behavior_key_to_behavior[coexisting_behavior_group[0]].sensorId

            # Find the overlapping time span of the co-existing behavior group
            max_timestamp = map_behavior_key_to_behavior[coexisting_behavior_group[0]].timestamp
            min_end = map_behavior_key_to_behavior[coexisting_behavior_group[0]].end
            for i in range(1, len(coexisting_behavior_group)):
                if max_timestamp < map_behavior_key_to_behavior[coexisting_behavior_group[i]].timestamp:
                    max_timestamp = map_behavior_key_to_behavior[coexisting_behavior_group[i]].timestamp
                if min_end > map_behavior_key_to_behavior[coexisting_behavior_group[i]].end:
                    min_end = map_behavior_key_to_behavior[coexisting_behavior_group[i]].end

            # Sample the locations of each behavior in the co-existing behavior group
            map_behavior_key_to_sampled_locations: Dict[str, List[List[float]]] = dict()
            for behavior_key in coexisting_behavior_group:
                sampled_locations = self._sample_overlapping_locations(map_behavior_key_to_behavior[behavior_key], max_timestamp, min_end)
                sampled_locations = [location for location in sampled_locations if location is not None]
                map_behavior_key_to_sampled_locations[behavior_key] = sampled_locations

            # Compute the mean locations of each cluster
            map_global_id_to_sampled_locations: Dict[int, List[List[float]]] = dict()
            for global_id in sorted_global_ids:
                aggregated_locations: List[List[float]] = list()
                for behavior_key in map_global_id_to_cluster[global_id]:
                    behavior  = map_behavior_key_to_behavior[behavior_key]
                    if behavior.sensorId == sensor_id:
                        continue
                    if behavior.timestamp > max_timestamp:
                        continue
                    if behavior.end < min_end:
                        continue
                    sampled_locations = self._sample_overlapping_locations(behavior, max_timestamp, min_end)
                    sampled_locations = [location for location in sampled_locations if location is not None]
                    if len(sampled_locations) == 0:
                        continue
                    aggregated_locations.append(sampled_locations)
                map_global_id_to_sampled_locations[global_id] = self._interpolate_and_average_locations(aggregated_locations)

            # Compute the spatio-temporal distance betweeen behaviors and clusters
            for behavior_key in coexisting_behavior_group:
                for global_id in sorted_global_ids:
                    spatio_temporal_dist = \
                        self._calculate_spatio_temporal_dist_between_trajectories(map_behavior_key_to_sampled_locations[behavior_key],
                                                                                  map_global_id_to_sampled_locations[global_id])
                    if (spatio_temporal_dist is not None) and \
                        ((max_spatio_temporal_dist is None) or (max_spatio_temporal_dist < spatio_temporal_dist)):
                        max_spatio_temporal_dist = spatio_temporal_dist
                    map_key_id_pair_to_dist[(behavior_key, global_id)] =  spatio_temporal_dist

            del map_behavior_key_to_sampled_locations
            del map_global_id_to_sampled_locations

        # Normalize spatio-temporal distance and combine it with appearance distance
        for key_id_pair in map_key_id_pair_to_dist.keys():
            if (max_spatio_temporal_dist is None) or (map_key_id_pair_to_dist[key_id_pair] is None):
                map_key_id_pair_to_dist[key_id_pair] = 1.
            else:
                map_key_id_pair_to_dist[key_id_pair] /= (max_spatio_temporal_dist + 1.)
            appearance_dist = 1.
            if key_id_pair in map_key_id_pair_to_appearance_dist:
                appearance_dist = map_key_id_pair_to_appearance_dist[key_id_pair]
            map_key_id_pair_to_dist[key_id_pair] = (map_key_id_pair_to_dist[key_id_pair] * self.config.clustering.spatioTemporalDistLambda) + \
                (appearance_dist * (1 - self.config.clustering.spatioTemporalDistLambda))

        # Build cost matrix
        cost_matrix: List[List[float]] = list()
        for behavior_key in coexisting_behavior_group:
            cost_matrix.append(list())
            for global_id in sorted_global_ids:
                cost_matrix[-1].append(map_key_id_pair_to_dist[(behavior_key, global_id)])
        del map_key_id_pair_to_dist

        # Apply Hungarian algorithm on the cost matrix for each group of co-existing behaviors
        if len(cost_matrix) == 1:
            min_cost = min(cost_matrix[0])
            if min_cost < self.config.clustering.reassignmentDistLooseThresh:
                map_behavior_key_to_global_ids[coexisting_behavior_group[0]].append(sorted_global_ids[cost_matrix[0].index(min_cost)])
        else:
            row_indices, col_indices = linear_sum_assignment(np.array(cost_matrix))
            for row, col in zip(row_indices, col_indices):
                if (cost_matrix[row][col] == min(cost_matrix[row])) and (cost_matrix[row][col] < self.config.clustering.reassignmentDistLooseThresh):
                    map_behavior_key_to_global_ids[coexisting_behavior_group[row]].append(sorted_global_ids[col])
        del cost_matrix

        return map_behavior_key_to_global_ids

    def _reassign_coexisting_behavior_groups(self, map_behavior_key_to_behavior: Dict[str, Behavior],
                                             map_global_id_to_cluster: Dict[int, List[str]],
                                             map_key_id_pair_to_appearance_dist: Dict[Tuple[str, int], float],
                                             coexisting_behavior_groups: List[List[str]]) -> \
                                                Dict[int, List[str]]:
        """
        Re-assigns co-existing behavior groups based on the Hungarian algorithm

        :param Dict[str,Behavior] map_behavior_key_to_behavior: map from behavior keys (sensor-object-timestamp IDs) to behavior objects
        :param Dict[int,List[str]] map_global_id_to_cluster: map from global IDs to clusters
        :param Dict[Tuple[str,int],float] map_key_id_pair_to_appearance_dist: map from pairs of behavior keys and global IDs to appearance distance
        :param List[List[str]] coexisting_behavior_groups: list of co-existing groups of behavior keys
        :return: map from global IDs to clusters
        :rtype: Dict[int,List[str]]
        """
        logging.info(f"Re-assigning co-existing behaviors to clusters...")

        sorted_global_ids = sorted(list(map_global_id_to_cluster))

        # num_processes = mp.cpu_count()
        # if num_processes > len(coexisting_behavior_groups):
        #     num_processes = len(coexisting_behavior_groups)

        # with mp.Pool(processes=num_processes) as pool:
        #     map_behavior_key_to_global_ids_list = pool.starmap(self._reassign_coexisting_behavior_group,
        #         [(coexisting_behavior_group, map_behavior_key_to_behavior, sorted_global_ids, map_global_id_to_cluster, map_key_id_pair_to_appearance_dist)
        #          for coexisting_behavior_group in coexisting_behavior_groups])

        map_behavior_key_to_global_ids_list: List[Dict[int, List[str]]] = list()
        for coexisting_behavior_group in coexisting_behavior_groups:
            map_behavior_key_to_global_ids_list.append(
                self._reassign_coexisting_behavior_group(coexisting_behavior_group, map_behavior_key_to_behavior, sorted_global_ids, map_global_id_to_cluster, map_key_id_pair_to_appearance_dist))

        del sorted_global_ids

        map_behavior_key_to_global_ids: Dict[str, List[int]] = defaultdict(list)
        for map_behavior_key_to_global_ids_instance in map_behavior_key_to_global_ids_list:
            for behavior_key, global_ids in map_behavior_key_to_global_ids_instance.items():
                map_behavior_key_to_global_ids[behavior_key].extend(global_ids)

        del map_behavior_key_to_global_ids_list

        # Map each behavior key to its most common global ID
        map_global_id_to_cluster: Dict[int, Set[str]] = defaultdict(set)
        for behavior_key in map_behavior_key_to_global_ids.keys():
            global_ids = map_behavior_key_to_global_ids[behavior_key]
            global_id_counter = Counter(global_ids).most_common(None)
            map_global_id_to_cluster[global_id_counter[0][0]].add(behavior_key)
        del map_behavior_key_to_global_ids

        return {global_id: self._sort_behavior_keys(list(cluster)) for global_id, cluster in map_global_id_to_cluster.items()}

    def _reassign_coexisting_behavior_groups_online(self, map_behavior_key_to_behavior: Dict[str, Behavior],
                                                    mtmc_state_objects_plus: Dict[str, MTMCStateObjectPlus],
                                                    map_key_id_pair_to_appearance_dist: Dict[Tuple[str, int], float],
                                                    coexisting_behavior_groups: List[List[str]],
                                                    matched_clusters: Dict[str, List[str]],
                                                    matched_behavior_key_set: Set[str]) -> \
                                                        Tuple[Dict[int, List[str]], List[List[str]]]:
        """
        Re-assigns co-existing behavior groups based on the Hungarian algorithm

        :param Dict[str, Behavior] map_behavior_key_to_behavior: map from behavior keys (sensor-object-timestamp IDs) to behavior objects
        :param Dict[str, MTMCStateObjectPlus] mtmc_state_objects_plus: map from global IDs to MTMC state objects plus
        :param Dict[Tuple[str,int],float]] map_key_id_pair_to_appearance_dist: map from pairs of behavior keys and global IDs to appearance distance
        :param List[List[str]] coexisting_behavior_groups: list of co-existing groups of behavior keys
        :param Dict[str,List[str]] matched_clusters: map from global IDs to matched behavior keys
        :param Set[str] matched_behavior_key_set: set of matched behaviors
        :return: map from global IDs to clusters and list of shadow clusters
        :rtype: Tuple[Dict[int,List[str]],List[List[str]]]
        """
        logging.info(f"Re-assigning co-existing behaviors to clusters...")

        sorted_global_ids = sorted([int(global_id) for global_id in mtmc_state_objects_plus.keys()])

        # Initialize the map from pairs of behavior keys and global IDs to distance
        map_key_id_pair_to_spatio_temporal_dist: Dict[Tuple[str, int], Optional[float]] = dict()
        for behavior_key in map_behavior_key_to_behavior.keys():
            for global_id in sorted_global_ids:
                map_key_id_pair_to_spatio_temporal_dist[behavior_key, global_id] = None

        # Compute the location and direction of each cluster
        map_global_id_to_loc_dir_vec: Dict[int, List[float]] = dict()
        for global_id in sorted_global_ids:
            global_id_str = str(global_id)
            aggregated_locations: List[List[float]] = list()
            sorted_timestamps = sorted(list(mtmc_state_objects_plus[global_id_str].locationsDict.keys()))
            for timestamp in sorted_timestamps:
                aggregated_locations.extend(mtmc_state_objects_plus[global_id_str].locationsDict[timestamp])
            if len(aggregated_locations) > 0:
                map_global_id_to_loc_dir_vec[global_id] = [(sum(coords)/len(coords)) for coords in zip(*aggregated_locations)]
                if len(aggregated_locations) > 1:
                    direction_vec = [(aggregated_locations[-1][1] - aggregated_locations[0][1]),
                                     (aggregated_locations[-1][0] - aggregated_locations[0][0])]
                    map_global_id_to_loc_dir_vec[global_id] += direction_vec
                    del direction_vec
            del aggregated_locations

        # Compute the location and direction of each behavior
        map_behavior_key_to_loc_dir_vec: Dict[str, List[Optional[float]]] = dict()
        for behavior_key in map_behavior_key_to_behavior.keys():
            if behavior_key in matched_behavior_key_set:
                continue

            num_locations = len(map_behavior_key_to_behavior[behavior_key].locations)
            if num_locations == 0:
                continue
            idx_location = num_locations - 1

            num_timestamps = len(map_behavior_key_to_behavior[behavior_key].timestamps)
            idx_timestamp = num_timestamps - 1

            # Only get the last locations for each behavior
            locations: List[List[float]] = list()
            timestamp_thresh = map_behavior_key_to_behavior[behavior_key].timestamps[-1] - timedelta(seconds=self.config.preprocessing.mtmcPlusRetentionInStateSec)
            while (idx_timestamp >= 0) and (map_behavior_key_to_behavior[behavior_key].timestamps[idx_timestamp] > timestamp_thresh):
                if map_behavior_key_to_behavior[behavior_key].locationMask[idx_timestamp]:
                    locations.append(map_behavior_key_to_behavior[behavior_key].locations[idx_location])
                    idx_location -= 1
                idx_timestamp -= 1
            if len(locations) > 0:
                map_behavior_key_to_loc_dir_vec[behavior_key] = [(sum(coords)/len(coords)) for coords in zip(*locations)]
                if len(locations) > 1:
                    direction_vec = [(locations[0][1] - locations[-1][1]), (locations[0][0] - locations[-1][0])]
                    map_behavior_key_to_loc_dir_vec[behavior_key] += direction_vec
                    del direction_vec
            del locations

        # Compute the spatio-temporal distance betweeen behaviors and clusters
        for behavior_key in map_behavior_key_to_behavior.keys():
            if behavior_key in matched_behavior_key_set:
                continue
            for global_id in sorted_global_ids:
                spatio_temporal_dist = None
                if (behavior_key in map_behavior_key_to_loc_dir_vec.keys()) and (global_id in map_global_id_to_loc_dir_vec.keys()):
                    spatio_temporal_dist = self._calculate_spatio_temporal_dist_between_points(map_behavior_key_to_loc_dir_vec[behavior_key],
                                                                                               map_global_id_to_loc_dir_vec[global_id])
                map_key_id_pair_to_spatio_temporal_dist[(behavior_key, global_id)] = spatio_temporal_dist

        del map_global_id_to_loc_dir_vec

        # Set appearance distances for matched behaviors
        for key_id_pair in map_key_id_pair_to_spatio_temporal_dist.keys():
            if key_id_pair[0] in matched_behavior_key_set:
                if key_id_pair[0] in matched_clusters[str(key_id_pair[1])]:
                    map_key_id_pair_to_appearance_dist[key_id_pair] = 0.
                    map_key_id_pair_to_spatio_temporal_dist[key_id_pair] = 0.
                else:
                    map_key_id_pair_to_appearance_dist[key_id_pair] = 1.
                    map_key_id_pair_to_spatio_temporal_dist[key_id_pair] = None

        map_behavior_key_to_global_ids_list: List[Dict[int, List[str]]] = list()
        map_matched_global_id_to_behavior_key_and_dist: Dict[int, Tuple[str, float]] = dict()
        map_behavior_key_to_coexisting_keys: Dict[str, Set[str]] = defaultdict(set)
        connected_behavior_key_set: Set[str] = set()
        for coexisting_behavior_group in coexisting_behavior_groups:
            map_behavior_key_to_global_ids: Dict[str, List[int]] = defaultdict(list)

            # Set maximum spatio-temporal distance
            max_spatio_temporal_dist = None
            coexisting_behavior_group_set = set(coexisting_behavior_group)
            for behavior_key in coexisting_behavior_group:
                map_behavior_key_to_coexisting_keys[behavior_key].update(coexisting_behavior_group_set)
                if behavior_key in matched_behavior_key_set:
                    continue
                for global_id in sorted_global_ids:
                    spatio_temporal_dist = map_key_id_pair_to_spatio_temporal_dist[(behavior_key, global_id)]
                    if spatio_temporal_dist is None:
                        continue
                    if (max_spatio_temporal_dist is None) or (spatio_temporal_dist > max_spatio_temporal_dist):
                        max_spatio_temporal_dist = spatio_temporal_dist
            del coexisting_behavior_group_set

            # Build cost matrix
            cost_matrix: List[List[float]] = list()
            for behavior_key in coexisting_behavior_group:
                cost_matrix.append(list())
                for global_id in sorted_global_ids:
                    key_id_pair = (behavior_key, global_id)
                    appearance_dist = map_key_id_pair_to_appearance_dist[key_id_pair]
                    spatio_temporal_dist = map_key_id_pair_to_spatio_temporal_dist[key_id_pair]
                    if self.config.clustering.enableOnlineSpatioTemporalConstraint:
                        # Apply hard constraint based on online spatio-temporal distance threshold
                        # Assume that all cameras have continuous coverage and behaviors far away will not be associated
                        if (spatio_temporal_dist is not None) and \
                            ((self.config.clustering.onlineSpatioTemporalDistThresh is None) or \
                             (spatio_temporal_dist < self.config.clustering.onlineSpatioTemporalDistThresh)):
                            if (max_spatio_temporal_dist is not None) and (max_spatio_temporal_dist > 0.):
                                spatio_temporal_dist /= max_spatio_temporal_dist
                            combined_dist = (spatio_temporal_dist * self.config.clustering.spatioTemporalDistLambda) + \
                                (appearance_dist * (1 - self.config.clustering.spatioTemporalDistLambda))
                            if 0. < combined_dist < self.config.clustering.reassignmentDistTightThresh:
                                if (global_id not in map_matched_global_id_to_behavior_key_and_dist.keys()) or \
                                    (combined_dist < map_matched_global_id_to_behavior_key_and_dist[global_id][1]):
                                    map_matched_global_id_to_behavior_key_and_dist[global_id] = (behavior_key, combined_dist)
                            cost_matrix[-1].append(combined_dist)
                        else:
                            cost_matrix[-1].append(1.)
                    else:
                        if (max_spatio_temporal_dist is None) or (spatio_temporal_dist is None):
                            spatio_temporal_dist = 1.
                        elif max_spatio_temporal_dist > 0.:
                            spatio_temporal_dist /= max_spatio_temporal_dist
                        cost_matrix[-1].append((spatio_temporal_dist * self.config.clustering.spatioTemporalDistLambda) + \
                            (appearance_dist * (1 - self.config.clustering.spatioTemporalDistLambda)))

            # Apply Hungarian algorithm on the cost matrix for each group of co-existing behaviors
            if len(cost_matrix) == 1:
                min_cost = min(cost_matrix[0])
                if min_cost < self.config.clustering.reassignmentDistLooseThresh:
                    map_behavior_key_to_global_ids[coexisting_behavior_group[0]].append(sorted_global_ids[cost_matrix[0].index(min_cost)])
                    connected_behavior_key_set.add(coexisting_behavior_group[0])
            else:
                row_indices, col_indices = linear_sum_assignment(np.array(cost_matrix))
                for row, col in zip(row_indices, col_indices):
                    if cost_matrix[row][col] < self.config.clustering.reassignmentDistLooseThresh:
                        connected_behavior_key_set.add(coexisting_behavior_group[row])
                        if cost_matrix[row][col] == min(cost_matrix[row]):
                            map_behavior_key_to_global_ids[coexisting_behavior_group[row]].append(sorted_global_ids[col])
            del cost_matrix

            map_behavior_key_to_global_ids_list.append(map_behavior_key_to_global_ids)

        del sorted_global_ids
        del map_key_id_pair_to_spatio_temporal_dist

        map_behavior_key_to_global_ids: Dict[str, List[int]] = defaultdict(list)
        for map_behavior_key_to_global_ids_instance in map_behavior_key_to_global_ids_list:
            for behavior_key, global_ids in map_behavior_key_to_global_ids_instance.items():
                map_behavior_key_to_global_ids[behavior_key].extend(global_ids)

        del map_behavior_key_to_global_ids_list

        # Update map from matched behavior keys to global IDs
        map_matched_behavior_key_to_global_id: Dict[str, int] = dict()
        for global_id, behavior_key_and_dist in map_matched_global_id_to_behavior_key_and_dist.items():
            map_matched_behavior_key_to_global_id[behavior_key_and_dist[0]] = global_id
        del map_matched_global_id_to_behavior_key_and_dist
        map_coexisting_behavior_key_to_global_ids: Dict[str, Set[int]] = defaultdict(set)
        for behavior_key in map_matched_behavior_key_to_global_id.keys():
            for coexisting_behavior_key in map_behavior_key_to_coexisting_keys[behavior_key]:
                if coexisting_behavior_key == behavior_key:
                    continue
                map_coexisting_behavior_key_to_global_ids[coexisting_behavior_key].add(map_matched_behavior_key_to_global_id[behavior_key])
        del map_behavior_key_to_coexisting_keys
        for behavior_key in map_matched_behavior_key_to_global_id.keys():
            map_behavior_key_to_global_ids[behavior_key] = [map_matched_behavior_key_to_global_id[behavior_key]]
        del map_matched_behavior_key_to_global_id
        behavior_keys_to_delete: List[str] = list()
        for behavior_key in map_behavior_key_to_global_ids.keys():
            if behavior_key not in map_coexisting_behavior_key_to_global_ids.keys():
                continue
            global_ids: List[int] = list()
            for global_id in map_behavior_key_to_global_ids[behavior_key]:
                if global_id in map_coexisting_behavior_key_to_global_ids[behavior_key]:
                    continue
                global_ids.append(global_id)
            if len(global_ids) > 0:
                map_behavior_key_to_global_ids[behavior_key] = global_ids
            else:
                behavior_keys_to_delete.append(behavior_key)
        del map_coexisting_behavior_key_to_global_ids
        for behavior_key in behavior_keys_to_delete:
            map_behavior_key_to_global_ids.pop(behavior_key, None)
        del behavior_keys_to_delete

        # Map each matched behavior key to global IDs
        map_matched_behavior_key_to_global_ids: Dict[str, List[int]] = defaultdict(list)
        if len(matched_behavior_key_set) > 0:
            for global_id, matched_behavior_keys in matched_clusters.items():
                for matched_behavior_key in matched_behavior_keys:
                    map_matched_behavior_key_to_global_ids[matched_behavior_key].append(int(global_id))

        # Map each behavior key to its most common global ID
        map_global_id_to_cluster: Dict[int, Set[str]] = defaultdict(set)
        for behavior_key in map_behavior_key_to_global_ids.keys():
            if behavior_key in matched_behavior_key_set:
                for global_id in map_matched_behavior_key_to_global_ids[behavior_key]:
                    map_global_id_to_cluster[global_id].add(behavior_key)
            else:
                global_ids = map_behavior_key_to_global_ids[behavior_key]
                global_id_counter = Counter(global_ids).most_common(None)
                map_global_id_to_cluster[global_id_counter[0][0]].add(behavior_key)
        del map_matched_behavior_key_to_global_ids
        map_global_id_to_cluster = {global_id: self._sort_behavior_keys(list(cluster)) for global_id, cluster in map_global_id_to_cluster.items()}

        if not self.config.clustering.enableOnlineDynamicUpdate:
            del map_behavior_key_to_loc_dir_vec
            return map_global_id_to_cluster, list()

        # Find the unmatched behavior keys
        unmatched_behavior_keys: List[str] = list()
        for behavior_key in (set(map_behavior_key_to_behavior.keys()) - set(map_behavior_key_to_global_ids.keys()) - connected_behavior_key_set):
            if behavior_key in map_behavior_key_to_loc_dir_vec.keys():
                unmatched_behavior_keys.append(behavior_key)
        del map_behavior_key_to_global_ids
        del connected_behavior_key_set

        if len(unmatched_behavior_keys) == 0:
            del map_behavior_key_to_loc_dir_vec
            return map_global_id_to_cluster, list()

        if len(unmatched_behavior_keys) == 1:
            del map_behavior_key_to_loc_dir_vec
            return map_global_id_to_cluster, [unmatched_behavior_keys]

        # Find matched pairs of unmatched behavior keys
        shadow_matched_behavior_key_pairs: List[List[str]] = list()
        num_unatched_behavior_keys = len(unmatched_behavior_keys)
        for i in range(num_unatched_behavior_keys - 1):
            behavior_key_a = unmatched_behavior_keys[i]
            for j in range(i + 1, num_unatched_behavior_keys):
                behavior_key_b = unmatched_behavior_keys[j]
                appearance_dist = self._calculate_euclidean_dist(np.array(map_behavior_key_to_behavior[behavior_key_a].embeddings[0]),
                                                                 np.array(map_behavior_key_to_behavior[behavior_key_b].embeddings[0]))
                if appearance_dist > self.config.clustering.dynamicUpdateAppearanceDistThresh:
                    continue
                if (behavior_key_a in map_behavior_key_to_loc_dir_vec.keys()) and (behavior_key_b in map_behavior_key_to_loc_dir_vec.keys()):
                    spatio_temporal_dist = self._calculate_spatio_temporal_dist_between_points(map_behavior_key_to_loc_dir_vec[behavior_key_a],
                                                                                               map_behavior_key_to_loc_dir_vec[behavior_key_b])
                    if spatio_temporal_dist < self.config.clustering.dynamicUpdateSpatioTemporalDistThresh:
                        shadow_matched_behavior_key_pairs.append([behavior_key_a, behavior_key_b])
        del map_behavior_key_to_loc_dir_vec

        # Group the unmatched behavior keys
        shadow_clusters: List[List[str]] = list()
        map_behavior_key_to_shadow_matches: Dict[str, List[str]] = defaultdict(list)
        visited_behavior_keys: Set[str] = set()
        for behavior_key_pair in shadow_matched_behavior_key_pairs:
            map_behavior_key_to_shadow_matches[behavior_key_pair[0]].append(behavior_key_pair[1])
            map_behavior_key_to_shadow_matches[behavior_key_pair[1]].append(behavior_key_pair[0])
        for behavior_key in unmatched_behavior_keys:
            if behavior_key not in visited_behavior_keys:
                if behavior_key in map_behavior_key_to_shadow_matches.keys():
                    shadow_match_connected_component: List[str] = list()
                    shadow_match_queue = deque([behavior_key])
                    while shadow_match_queue:
                        shadow_match_node = shadow_match_queue.popleft()
                        if shadow_match_node not in visited_behavior_keys:
                            visited_behavior_keys.add(shadow_match_node)
                            shadow_match_connected_component.append(shadow_match_node)
                            for shadow_match_neighbor in map_behavior_key_to_shadow_matches[shadow_match_node]:
                                if shadow_match_neighbor not in visited_behavior_keys:
                                    shadow_match_queue.append(shadow_match_neighbor)
                    del shadow_match_queue
                    shadow_clusters.append(shadow_match_connected_component)
                    del shadow_match_connected_component
                else:
                    visited_behavior_keys.add(behavior_key)
                    shadow_clusters.append([behavior_key])
        del unmatched_behavior_keys
        del shadow_matched_behavior_key_pairs
        del map_behavior_key_to_shadow_matches
        del visited_behavior_keys

        return map_global_id_to_cluster, shadow_clusters

    def _suppress_overlapping_behaviors(self, behavior_keys: List[str], map_global_id_to_cluster: Dict[int, List[str]],
                                        coexisting_behavior_groups: List[List[str]],
                                        map_key_id_pair_to_appearance_dist: Dict[Tuple[str, int], float]) -> \
                                            Dict[int, List[str]]:
        """
        Suppresses overlapping behaviors using Linear Programming

        :param List[str] behavior_keys: list of behavior keys (sensor-object-timestamp IDs)
        :param Dict[int,List[str]] map_global_id_to_cluster: map from global IDs to clusters
        :param List[List[str]] coexisting_behavior_groups: list of co-existing groups of behavior keys
        :param Dict[Tuple[str,int],float] map_key_id_pair_to_appearance_dist: map from pairs of behavior keys and global IDs to appearance distance
        :return: map from global IDs to clusters
        :rtype: Dict[int,List[str]]
        """
        logging.info(f"Suppressing overlapping behaviors...")

        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # Define variables indicating whether a behavior key is in a cluster
        assign_vars = {key_id_pair: solver.IntVar(0, 1, 'assign_%s_%s' % key_id_pair) for key_id_pair in map_key_id_pair_to_appearance_dist}

        # Define objective function
        objective = solver.Objective()
        for key_id_pair, assign_var in assign_vars.items():
            objective.SetCoefficient(assign_var, map_key_id_pair_to_appearance_dist[key_id_pair])
        objective.SetMinimization()

        # Create constraints ensuring each behavior key is assigned to one cluster only
        sorted_global_ids = sorted(map_global_id_to_cluster.keys())
        for behavior_key in behavior_keys:
            solver.Add(solver.Sum([assign_vars[(behavior_key, global_id)] for global_id in sorted_global_ids if (behavior_key, global_id) in assign_vars]) == 1)

        # Create constraints ensuring that behavior keys in the co-existing behavior groups are not in the same cluster
        for coexisting_behavior_group in coexisting_behavior_groups:
            for global_id in sorted_global_ids:
                solver.Add(solver.Sum([assign_vars[(behavior_key, global_id)] for behavior_key in coexisting_behavior_group if (behavior_key, global_id) in assign_vars]) <= 1)

        # Solve the problem
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            updated_clusters = {global_id: [] for global_id in sorted_global_ids}
            for key_id_pair, assign_var in assign_vars.items():
                if assign_var.solution_value() > 0:
                    updated_clusters[key_id_pair[1]].append(key_id_pair[0])

            return updated_clusters

        else:
            logging.info("An optimal solution for suppressing overlapping behaviors cannot be found.")
            return map_global_id_to_cluster

    def _create_mtmc_objects(self, map_behavior_key_to_behavior: Dict[str, Behavior], map_global_id_to_cluster: Dict[int, List[str]]) -> List[MTMCObject]:
        """
        Creates MTMC objects from behaviors and clusters

        :param Dict[str,Behavior] map_behavior_key_to_behavior: map from behavior keys (sensor-object-timestamp IDs) to behavior objects
        :param Dict[int,List[str]] map_global_id_to_cluster: map from global IDs to clusters
        :return: list of MTMC objects
        :rtype: List[MTMCObject]
        """
        logging.info(f"Creating MTMC objects...")

        mtmc_objects: List[MTMCObject] = list()

        for global_id in sorted(map_global_id_to_cluster.keys()):
            if len(map_global_id_to_cluster[global_id]) == 0:
                continue 
            batch_id = self.config.io.batchId
            object_type = None
            mtmc_timestamp = map_behavior_key_to_behavior[map_global_id_to_cluster[global_id][0]].timestamp
            mtmc_end = map_behavior_key_to_behavior[map_global_id_to_cluster[global_id][0]].end
            matched: List[Behavior] = list()

            for behavior_key in map_global_id_to_cluster[global_id]:
                if object_type is None:
                    object_type = map_behavior_key_to_behavior[behavior_key].objectType
                elif object_type != map_behavior_key_to_behavior[behavior_key].objectType:
                    logging.error(f"ERROR: Unmatched object type {map_behavior_key_to_behavior[behavior_key].objectType} with {object_type} in the same cluster.")
                    exit(1)

                matched_behavior = map_behavior_key_to_behavior[behavior_key].copy()
                matched_behavior.timestamps = None
                matched_behavior.frameIds = None
                matched_behavior.bboxes = None
                matched_behavior.confidences = None
                matched_behavior.locations = None
                matched_behavior.locationMask = None
                matched_behavior.embeddings = None
                matched.append(matched_behavior)

                timestamp = matched_behavior.timestamp
                if mtmc_timestamp > timestamp:
                    mtmc_timestamp = timestamp

                end = matched_behavior.end
                if mtmc_end < end:
                    mtmc_end = end

            matched = sorted(matched, key=lambda matched_behavior: matched_behavior.timestamp)

            mtmc_object = MTMCObject(batchId=batch_id, globalId=str(global_id), objectType=object_type,
                                     timestamp=mtmc_timestamp, end=mtmc_end, matched=matched)
            mtmc_objects.append(mtmc_object)

        return mtmc_objects

    def cluster(self, behaviors: List[Behavior]) -> Tuple[List[MTMCObject], List[str], np.array]:
        """
        Clusters behaviors to get MTMC objects

        :param List[Behavior] behaviors: list of behaviors
        :return: list of MTMC objects, list of behavior keys, and embedding array
        :rtype: Tuple[List[MTMCObject],List[str],np.array]
        ::

            mtmc_objects, behavior_keys, embedding_array = clusterer.cluster(behaviors)
        """
        map_behavior_key_to_behavior: Dict[str, Behavior] = dict()
        for behavior in behaviors:
            map_behavior_key_to_behavior[behavior.key] = behavior
        del behaviors

        behavior_keys = list(map_behavior_key_to_behavior.keys())
        behavior_keys = self._sort_behavior_keys(behavior_keys)

        # Combine all embedding features
        embeddings: List[List[float]] = list()
        for behavior_key in behavior_keys:
            embeddings.extend(map_behavior_key_to_behavior[behavior_key].embeddings)
        embedding_array = np.array(embeddings)
        del embeddings

        if len(behavior_keys) == 0:
            logging.info("No. clusters: 0")
            return list(), behavior_keys, embedding_array
        elif len(behavior_keys) == 1:
            logging.info("No. clusters: 1")
            return self._create_mtmc_objects(map_behavior_key_to_behavior, {1: behavior_keys}), behavior_keys, embedding_array

        # Count clusters for agglomerative clustering
        num_clusters = self.config.clustering.overwrittenNumClusters
        if self.config.clustering.clusteringAlgo == "AgglomerativeClustering":
            if num_clusters is None:
                num_clusters = self._count_clusters(embedding_array)
            elif num_clusters > embedding_array.shape[0]:
                num_clusters = embedding_array.shape[0]

        # Find behaviors that co-exist at the same time within the same sensor
        coexisting_behavior_groups = None
        if self.config.clustering.numReassignmentIterations > 0:
            coexisting_behavior_groups = self._find_coexisting_behaviors(map_behavior_key_to_behavior, behavior_keys)
            if (self.config.clustering.clusteringAlgo == "AgglomerativeClustering") and (len(coexisting_behavior_groups) > 0):
                max_num_coexisting_behaviors = max(len(coexisting_behavior_group) for coexisting_behavior_group in coexisting_behavior_groups)
                if num_clusters < max_num_coexisting_behaviors:
                    num_clusters = max_num_coexisting_behaviors
                    logging.info(f"No. clusters updated by max co-existing behaviors: {num_clusters}")

        # Cluster embeddings
        map_global_id_to_cluster = self._cluster_embeddings(behavior_keys, embedding_array, num_clusters)

        map_key_id_pair_to_appearance_dist: Dict[Tuple[str, int], float] = dict()
        if (len(map_global_id_to_cluster) > 0) and (self.config.clustering.numReassignmentIterations > 0) and (len(coexisting_behavior_groups) > 0):
            for i in range(self.config.clustering.numReassignmentIterations):
                logging.info(f"Iteration #{i} for re-assignment...")

                if (self.config.clustering.spatioTemporalDistLambda < 1) or self.config.clustering.suppressOverlappingBehaviors:
                    # Compute mean embedding of each cluster
                    map_global_id_to_mean_embedding = self._compute_mean_embeddings(behavior_keys, embedding_array, map_global_id_to_cluster)

                    # Calculate appearance distances between behaviors and the mean embeddings of clusters
                    map_key_id_pair_to_appearance_dist = self._calculate_appearance_dists(map_behavior_key_to_behavior, map_global_id_to_mean_embedding, set())
                    del map_global_id_to_mean_embedding

                # Re-assign co-existing behaviors to clusters based on the Hungarian algorithm
                map_global_id_to_cluster = self._reassign_coexisting_behavior_groups(map_behavior_key_to_behavior, map_global_id_to_cluster, map_key_id_pair_to_appearance_dist, coexisting_behavior_groups)

            # Suppress overlapping behaviors using Linear Programming
            if (len(map_global_id_to_cluster) > 0) and self.config.clustering.suppressOverlappingBehaviors:
                map_global_id_to_cluster = self._suppress_overlapping_behaviors(behavior_keys, map_global_id_to_cluster, coexisting_behavior_groups, map_key_id_pair_to_appearance_dist)

        del coexisting_behavior_groups
        del map_key_id_pair_to_appearance_dist

        # Create MTMC objects
        mtmc_objects = self._create_mtmc_objects(map_behavior_key_to_behavior, map_global_id_to_cluster)
        del map_behavior_key_to_behavior
        del map_global_id_to_cluster

        return mtmc_objects, behavior_keys, embedding_array

    def match_online(self, behaviors: List[Behavior], mtmc_state_objects_plus: Dict[str, MTMCStateObjectPlus]) -> \
        Tuple[List[MTMCObject], Dict[str, List[List[float]]]]:
        """
        Matches behaviors in an online mode

        :param List[Behavior] behaviors: list of behaviors
        :param Dict[str,MTMCStateObjectPlus] mtmc_state_objects_plus: map from global IDs to MTMC state objects plus
        :return: list of MTMC objects and map from global IDs to mean embeddings
        :rtype: Tuple[List[MTMCObject],Dict[str,List[List[float]]]]
        ::

            mtmc_objects, map_global_id_to_mean_embedding = clusterer.match_online(behaviors, mtmc_state_objects_plus)
        """
        behavior_keys: List[str] = list()
        map_behavior_key_to_behavior: Dict[str, Behavior] = dict()
        for behavior in behaviors:
            behavior_key = behavior.key
            behavior_keys.append(behavior_key)
            map_behavior_key_to_behavior[behavior_key] = behavior
        del behaviors

        if len(behavior_keys) == 0:
            logging.info(f"Returning MTMC objects directly for empty behavior list...")
            del behavior_keys
            del map_behavior_key_to_behavior
            return list(), {global_id: mtmc_state_object_plus.embeddings for global_id, mtmc_state_object_plus in mtmc_state_objects_plus.items()}

        # Sort behavior keys
        behavior_keys = self._sort_behavior_keys(behavior_keys)

        # Combine all embedding features
        embeddings: List[List[float]] = list()
        for behavior_key in behavior_keys:
            embeddings.extend(map_behavior_key_to_behavior[behavior_key].embeddings)
        embedding_array = np.array(embeddings)
        del embeddings

        matched_behavior_key_set: Set[str] = set()
        matched_clusters: Dict[str, List[str]] = dict()
        if self.config.clustering.skipAssignedBehaviors:
            behavior_key_set = set(behavior_keys)

            # Only consider behavior keys that are not matched
            for global_id, mtmc_state_object_plus in mtmc_state_objects_plus.items():
                matched_clusters[global_id] = list()
                for matched_behavior_key in mtmc_state_object_plus.matchedBehaviorKeys:
                    if matched_behavior_key in behavior_key_set:
                        matched_clusters[global_id].append(matched_behavior_key)
                        matched_behavior_key_set.add(matched_behavior_key)

            # If all behaviors have already been matched, return directly
            if len(behavior_key_set - matched_behavior_key_set) == 0:
                logging.info(f"Returning MTMC objects directly for all behaviors have been matched...")
                del behavior_key_set
                del matched_behavior_key_set
                map_global_id_to_mean_embedding = self._compute_mean_embeddings(behavior_keys, embedding_array, matched_clusters)
                del behavior_keys
                del embedding_array
                map_global_id_to_mean_embedding = {str(global_id): [mean_embedding_array.tolist()] for global_id, mean_embedding_array in map_global_id_to_mean_embedding.items()}
                mtmc_objects = self._create_mtmc_objects(map_behavior_key_to_behavior, matched_clusters)
                del map_behavior_key_to_behavior
                del matched_clusters
                return mtmc_objects, map_global_id_to_mean_embedding
            del behavior_key_set

        # Convert mean embeddings to expected format
        map_global_id_to_mean_embedding = {int(global_id): np.array(mtmc_state_object_plus.embeddings[0])
                                           for global_id, mtmc_state_object_plus in mtmc_state_objects_plus.items()}

        # Find behaviors that co-exist at the same time within the same sensor
        coexisting_behavior_groups = self._find_coexisting_behaviors(map_behavior_key_to_behavior, behavior_keys)

        map_global_id_to_cluster: Dict[int, List[str]] = dict()
        shadow_clusters: List[List[str]] = list()
        if len(coexisting_behavior_groups) > 0:
            # Calculate appearance distances between behaviors and the mean embeddings of clusters
            map_key_id_pair_to_appearance_dist = self._calculate_appearance_dists(map_behavior_key_to_behavior, map_global_id_to_mean_embedding, matched_behavior_key_set)

            # Re-assign co-existing behaviors to clusters based on the Hungarian algorithm
            map_global_id_to_cluster, shadow_clusters = self._reassign_coexisting_behavior_groups_online(map_behavior_key_to_behavior, mtmc_state_objects_plus, map_key_id_pair_to_appearance_dist,
                                                                                                         coexisting_behavior_groups, matched_clusters, matched_behavior_key_set)
            del matched_clusters
            del matched_behavior_key_set

            # Suppress overlapping behaviors using Linear Programming
            if self.config.clustering.suppressOverlappingBehaviors:
                map_global_id_to_cluster = self._suppress_overlapping_behaviors(behavior_keys, map_global_id_to_cluster, coexisting_behavior_groups, map_key_id_pair_to_appearance_dist)
            del map_key_id_pair_to_appearance_dist

            # Combine with shadow clusters
            min_global_id = min(map_global_id_to_cluster.keys(), default=0)
            for i in range(len(shadow_clusters)):
                min_global_id -= 1
                map_global_id_to_cluster[min_global_id] = shadow_clusters[i]

        del coexisting_behavior_groups

        # Compute mean embeddings
        map_global_id_to_mean_embedding = self._compute_mean_embeddings(behavior_keys, embedding_array, map_global_id_to_cluster)
        del behavior_keys
        del embedding_array
        map_global_id_to_mean_embedding = {str(global_id): [mean_embedding_array.tolist()] for global_id, mean_embedding_array in map_global_id_to_mean_embedding.items()}

        # Create MTMC objects
        mtmc_objects = self._create_mtmc_objects(map_behavior_key_to_behavior, map_global_id_to_cluster)
        del map_behavior_key_to_behavior
        del map_global_id_to_cluster

        return mtmc_objects, map_global_id_to_mean_embedding

    def stitch_mtmc_objects_with_state(self, mtmc_objects: List[MTMCObject], mtmc_state_objects_plus: Dict[str, MTMCStateObjectPlus]) ->  List[MTMCObject]:
        """
        Stitches MTMC objects with existing MTMC state objects plus

        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :param Dict[str,MTMCStateObjectPlus] mtmc_state_objects_plus: map from global IDs to MTMC state objects plus
        :return: updated list of MTMC objects
        :rtype: List[MTMCObject]
        ::

            mtmc_objects = clusterer.stitch_mtmc_objects_with_state(mtmc_objects, mtmc_state_objects_plus)
        """
        # Map each global ID to behavior keys for MTMC state objects plus
        map_global_id_to_behavior_keys_in_state: Dict[int, Set[str]] = dict()
        for global_id in mtmc_state_objects_plus.keys():
            map_global_id_to_behavior_keys_in_state[int(global_id)] = mtmc_state_objects_plus[global_id].matchedBehaviorKeys
        sorted_global_id_and_behavior_keys_in_state = sorted(map_global_id_to_behavior_keys_in_state.items(), key=lambda item: len(item[1]), reverse=True)

        # Map each global ID to behavior keys for MTMC objects
        map_global_id_to_behavior_keys: Dict[int, Set[str]] = dict()
        for mtmc_object in mtmc_objects:
            global_id = int(mtmc_object.globalId)
            map_global_id_to_behavior_keys[global_id] = set()
            for matched_behavior in mtmc_object.matched:
                map_global_id_to_behavior_keys[global_id].add(matched_behavior.key)
        sorted_global_id_and_behavior_keys = sorted(map_global_id_to_behavior_keys.items(), key=lambda item: len(item[1]), reverse=True)
        del map_global_id_to_behavior_keys

        # Update global IDs of MTMC objects
        max_global_id = max(map_global_id_to_behavior_keys_in_state.keys(), default=0)
        del map_global_id_to_behavior_keys_in_state
        map_global_id_to_global_id_in_state: Dict[int, int] = dict()
        used_global_ids_in_state: Set[int] = set()
        for global_id, behavior_keys in sorted_global_id_and_behavior_keys:
            max_overlap = 0
            best_match = None
            for global_id_in_state, behavior_keys_in_state in sorted_global_id_and_behavior_keys_in_state:
                if global_id_in_state in used_global_ids_in_state:
                    continue
                overlap = len(behavior_keys & behavior_keys_in_state)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = global_id_in_state
            if (max_overlap > 0) and (best_match >= 0):
                map_global_id_to_global_id_in_state[global_id] = best_match
                used_global_ids_in_state.add(best_match)
            else:
                max_global_id += 1
                map_global_id_to_global_id_in_state[global_id] = max_global_id
        del sorted_global_id_and_behavior_keys_in_state
        del sorted_global_id_and_behavior_keys
        del used_global_ids_in_state

        for i in range(len(mtmc_objects)):
            mtmc_objects[i].globalId = str(map_global_id_to_global_id_in_state[int(mtmc_objects[i].globalId)])
        del map_global_id_to_global_id_in_state

        return mtmc_objects
