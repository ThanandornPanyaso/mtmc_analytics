# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
import numpy as np
from typing import Dict, List, Set
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from mdx.mtmc.config import AppConfig
from mdx.mtmc.schema import Behavior, MTMCObject, SensorStateObject, MTMCObjectPlusCount, \
    MTMCObjectPlusLocations, MTMCObjectsPlus, MTMCStateObjectPlus
from mdx.mtmc.core.data import calculate_bbox_area, calculate_bbox_aspect_ratio, normalize_vector


class StateManager:
    """
    Module to manage MTMC plus state

    :param AppConfig config: configuration for the app
    ::

        mtmc_plus_state = StateManager(config)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self._state: Dict[str, MTMCStateObjectPlus] = dict()
        self._max_global_id = -1
        self._ratio_assigned_behavior_keys = None
        self.sensor_state_objects: Dict[str, SensorStateObject] = dict()

    def get_num_mtmc_state_objects_plus(self) -> int:
        """
        Gets the number of MTMC state objects plus

        :return: number of MTMC state objects plus
        :rtype: int
        ::

            num_mtmc_state_objects_plus = mtmc_plus_state.get_num_mtmc_state_objects_plus()
        """
        return len(self._state.keys())

    def get_mtmc_state_objects_plus(self) -> Dict[str, MTMCStateObjectPlus]:
        """
        Gets MTMC state objects plus

        :return: MTMC state objects plus
        :rtype: Dict[str,MTMCStateObjectPlus]
        ::

            mtmc_state_objects_plus = mtmc_plus_state.get_mtmc_state_objects_plus()
        """
        return self._state

    def set_sensor_state_objects(self, sensor_state_objects: Dict[str, SensorStateObject]) -> None:
        """
        Sets sensor state objects

        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :return: None
        ::

            mtmc_plus_state.set_sensor_state_objects(sensor_state_objects)
        """
        self.sensor_state_objects = sensor_state_objects

    def check_init_state(self, behaviors: List[Behavior]) -> bool:
        """
        Checks whether to (re-)initialize the state

        :param List[Behavior] mtmc_objects: list of behaviors
        :return: flag indicating that the state needs to be (re-)initialized
        :rtype: bool
        ::

            flag_init_state = mtmc_plus_state.check_init_state(behaviors)
        """
        num_mtmc_state_objects_plus = 0
        for global_id in self._state.keys():
            if int(global_id) >= 0:
                num_mtmc_state_objects_plus += 1

        if num_mtmc_state_objects_plus == 0:
            logging.info(f"Initializing MTMC plus state due to empty state...")
            return True

        overwritten_num_clusters = self.config.clustering.overwrittenNumClusters
        mtmc_plus_reinit_diff_ratio_clusters = self.config.preprocessing.mtmcPlusReinitDiffRatioClusters
        if (overwritten_num_clusters is not None) and (mtmc_plus_reinit_diff_ratio_clusters is not None):
            if abs(overwritten_num_clusters - num_mtmc_state_objects_plus) > (overwritten_num_clusters * mtmc_plus_reinit_diff_ratio_clusters):
                logging.info(f"Initializing MTMC plus state due to difference between the number of clusters {num_mtmc_state_objects_plus} and overwritten number is large...")
                return True

        live_behavior_key_set: Set[str] = set()
        for behavior in behaviors:
            live_behavior_key_set.add(behavior.key)

        state_behavior_key_set: Set[str] = set()
        for global_id in self._state.keys():
            state_behavior_key_set.update(self._state[global_id].matchedBehaviorKeys)

        if (self._ratio_assigned_behavior_keys is not None) and (self._ratio_assigned_behavior_keys < self.config.preprocessing.mtmcPlusReinitRatioAssignedBehaviors):
            logging.info(f"Initializing MTMC plus state due to the ratio of assigned behaviors {self._ratio_assigned_behavior_keys} is small...")
            return True
        del live_behavior_key_set
        del state_behavior_key_set

        logging.info(f"Running online tracking based on Hungarian matching...")
        return False

    def check_init_buffer_ready(self, behaviors: List[Behavior]) -> bool:
        """
        Checks whether the buffer of live behaviors for initialization is sufficiently long

        :param List[Behavior] behaviors: list of behaviors
        :return: flag indicating that the buffer of live behaviors for initialization is sufficiently long
        :rtype: bool
        ::

            flag_init_buffer_ready = mtmc_plus_state.check_init_buffer_len(behaviors)
        """
        min_timestamp = None
        max_end = None
        for behavior in behaviors:
            if (min_timestamp is None) or (min_timestamp > behavior.timestamp):
                min_timestamp = behavior.timestamp
            if (max_end is None) or (max_end < behavior.end):
                max_end = behavior.end

        if (min_timestamp is None) or (max_end is None):
            return False

        return max_end > (min_timestamp + timedelta(seconds=self.config.preprocessing.mtmcPlusInitBufferLenSec))

    def compute_mean_embeddings(self, mtmc_objects: List[MTMCObject], behavior_keys: List[str], embedding_array: np.array) -> \
        Dict[str, List[List[float]]]:
        """
        Computes mean embeddings

        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :param List[str] behavior_keys: list of behvaior keys
        :param np.array embedding_array: embedding array
        :return: map from global IDs to mean embeddings
        :rtype: Dict[str,List[List[float]]]
        ::

            map_global_id_to_mean_embedding = mtmc_plus_state.compute_mean_embeddings(mtmc_objects, behavior_keys, embedding_array)
        """
        # Map behavior keys to global IDs of clusters
        map_behavior_key_to_global_ids: Dict[str, List[str]] = defaultdict(list)
        for mtmc_object in mtmc_objects:
            for behavior in mtmc_object.matched:
                map_behavior_key_to_global_ids[behavior.key].append(mtmc_object.globalId)

        # Group embeddings for each cluster
        map_global_id_to_mean_embedding: Dict[str, List[List[float]]] = defaultdict(list)
        for i in range(len(behavior_keys)):
            for global_id in map_behavior_key_to_global_ids[behavior_keys[i]]:
                map_global_id_to_mean_embedding[global_id].append(embedding_array[i])
        del map_behavior_key_to_global_ids

        # Compute mean embedding for each cluster
        for global_id in map_global_id_to_mean_embedding.keys():
            map_global_id_to_mean_embedding[global_id] = np.sum(map_global_id_to_mean_embedding[global_id], axis=0)
            map_global_id_to_mean_embedding[global_id] = normalize_vector(map_global_id_to_mean_embedding[global_id])
            if map_global_id_to_mean_embedding[global_id] is not None:
                map_global_id_to_mean_embedding[global_id] = [map_global_id_to_mean_embedding[global_id].tolist()]
            else:
                map_global_id_to_mean_embedding.pop(global_id, None)

        return map_global_id_to_mean_embedding

    def update_mean_embeddings(self, map_global_id_to_mean_embedding: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
        """
        Updates mean embeddings

        :param Dict[str,List[List[float]]] map_global_id_to_mean_embedding: map from global IDs to mean embeddings
        :return: updated map from global IDs to mean embeddings
        :rtype: Dict[str,List[List[float]]]
        ::

            updated_map_global_id_to_mean_embedding = mtmc_plus_state.update_mean_embeddings(map_global_id_to_mean_embedding)
        """
        updated_map_global_id_to_mean_embedding: Dict[str, List[List[float]]] = dict()

        for global_id in self._state.keys():
            if global_id not in map_global_id_to_mean_embedding.keys():
                updated_map_global_id_to_mean_embedding[global_id] = self._state[global_id].embeddings
                continue
            updated_map_global_id_to_mean_embedding[global_id] = (np.array(self._state[global_id].embeddings[0]) * (1 - self.config.clustering.meanEmbeddingsUpdateRate)) + \
                (np.array(map_global_id_to_mean_embedding[global_id][0]) * self.config.clustering.meanEmbeddingsUpdateRate)
            updated_map_global_id_to_mean_embedding[global_id] = normalize_vector(updated_map_global_id_to_mean_embedding[global_id])
            if updated_map_global_id_to_mean_embedding[global_id] is not None:
                updated_map_global_id_to_mean_embedding[global_id] = [updated_map_global_id_to_mean_embedding[global_id].tolist()]
            else:
                updated_map_global_id_to_mean_embedding.pop(global_id, None)

        # Add shadow clusters
        if self.config.clustering.enableOnlineDynamicUpdate:
            for global_id in map_global_id_to_mean_embedding.keys():
                if global_id not in updated_map_global_id_to_mean_embedding.keys():
                    updated_map_global_id_to_mean_embedding[global_id] = map_global_id_to_mean_embedding[global_id]

        return updated_map_global_id_to_mean_embedding

    def get_mtmc_objects_plus(self, mtmc_objects: List[MTMCObject], behaviors: List[Behavior]) -> MTMCObjectsPlus:
        """
        Gets MTMC objects plus locations

        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :param List[Behavior] behaviors: list of behvaiors
        :return: MTMC objects plus locations
        :rtype: MTMCObjectsPlus
        ::

            mtmc_objects_plus = mtmc_plus_state.get_mtmc_objects_plus(mtmc_objects, behaviors)
        """
        # Create behavior dictionary and find the time interval of the current batch
        behavior_dict: Dict[str, List[Behavior]] = defaultdict(list)
        max_end = behaviors[0].end
        max_end_frame = int(behaviors[0].endFrame)
        for behavior in behaviors:
            behavior_dict[behavior.id].append(behavior)
            end = behavior.end
            if end > max_end:
                max_end = end
            end_frame = int(behavior.endFrame)
            if end_frame > max_end_frame:
                max_end_frame = end_frame

        # Initialize MTMC objects plus
        mtmc_objects_plus = MTMCObjectsPlus(place=behaviors[0].place,
                                            timestamp=max_end,
                                            frameId=str(max_end_frame),
                                            objectCounts=list(),
                                            locationsOfObjects=list())

        # Create individual MTMC object plus locations
        map_object_type_to_count: Dict[str, int] = dict()
        assigned_behavior_keys: Set[str] = set()
        for mtmc_object in mtmc_objects:
            global_id = mtmc_object.globalId
            object_type = mtmc_object.objectType
            locations: List[List[float]] = list()
            weights: List[float] = list()
            matched_behavior_keys: Set[str] = set()
            for matched_behavior in mtmc_object.matched:
                matched_behavior_id = matched_behavior.id
                if matched_behavior_id not in behavior_dict.keys():
                    continue
                matched_behavior_keys.add(matched_behavior.key)
                for behavior in behavior_dict[matched_behavior_id]:
                    idx_location = 0
                    sensor_id = behavior.sensorId
                    frame_width = None
                    frame_height = None
                    timestamp_thresh = max_end - timedelta(seconds=self.config.streaming.mtmcPlusLocationWindowSec)
                    if sensor_id in self.sensor_state_objects.keys():
                        frame_width = self.sensor_state_objects[sensor_id].frameWidth
                        frame_height = self.sensor_state_objects[sensor_id].frameHeight
                    for i in range(len(behavior.timestamps)):
                        if behavior.locationMask[i]:
                            if (behavior.timestamps[i] > timestamp_thresh) and \
                                ((behavior.timestamps[i] >= matched_behavior.timestamp) and 
                                 (behavior.timestamps[i] <= matched_behavior.end)):
                                locations.append(behavior.locations[idx_location])
                                if (frame_width is not None) and (frame_height is not None):
                                    weights.append((calculate_bbox_area(behavior.bboxes[i]) / (frame_width * frame_height)) /
                                                   calculate_bbox_aspect_ratio(behavior.bboxes[i]))
                                else:
                                    weights.append(1. / calculate_bbox_aspect_ratio(behavior.bboxes[i]))
                            idx_location += 1

            if int(global_id) >= 0:
                assigned_behavior_keys.update(matched_behavior_keys)

            if len(locations) > 0:
                sum_x_coords = 0.0
                sum_y_coords = 0.0
                for i in range(len(locations)):
                    sum_x_coords += locations[i][0] * weights[i]
                    sum_y_coords += locations[i][1] * weights[i]
                sum_weights = sum(weights)
                locations = [[(sum_x_coords / sum_weights), (sum_y_coords / sum_weights)]]
                mtmc_object_plus = MTMCObjectPlusLocations(id=global_id, type=object_type,
                                                           locations=locations, matchedBehaviorKeys=matched_behavior_keys)
                mtmc_objects_plus.locationsOfObjects.append(mtmc_object_plus)

                if int(mtmc_object.globalId) >= 0: 
                    if object_type not in map_object_type_to_count.keys():
                        map_object_type_to_count[object_type] = 0
                    map_object_type_to_count[object_type] += 1

        # Update object counts of MTMC objects
        for object_type in map_object_type_to_count.keys():
            mtmc_objects_plus.objectCounts.append(MTMCObjectPlusCount(type=object_type, count=map_object_type_to_count[object_type]))

        # Update ratio of assigned behavior keys
        self._ratio_assigned_behavior_keys = float(len(assigned_behavior_keys)) / float(len(behaviors))

        return mtmc_objects_plus

    def update_mtmc_objects_plus(self, mtmc_objects_plus: MTMCObjectsPlus, map_global_id_to_mean_embedding: Dict[str, List[List[float]]]) -> MTMCObjectsPlus:
        """
        Update MTMC objects plus with smoothed locations

        :param MTMCObjectsPlus mtmc_objects_plus: MTMC objects plus locations
        :param Dict[str,List[List[float]]] map_global_id_to_mean_embedding: map from global IDs to mean embeddings
        :return: updated MTMC objects plus locations
        :rtype: MTMCObjectsPlus
        ::

            updated_mtmc_objects_plus = mtmc_plus_state.update_mtmc_objects_plus(mtmc_objects_plus, map_global_id_to_mean_embedding)
        """
        locations_of_objects: List[MTMCObjectPlusLocations] = list()
        for i in range(len(mtmc_objects_plus.locationsOfObjects)):
            # Update MTMC plus state
            global_id = mtmc_objects_plus.locationsOfObjects[i].id
            timestamp = mtmc_objects_plus.timestamp
            object_type = mtmc_objects_plus.locationsOfObjects[i].type
            locations = mtmc_objects_plus.locationsOfObjects[i].locations
            matched_behavior_keys = mtmc_objects_plus.locationsOfObjects[i].matchedBehaviorKeys
            if global_id not in self._state.keys():
                if global_id not in map_global_id_to_mean_embedding.keys():
                    continue
                self._state[global_id] = MTMCStateObjectPlus(id=global_id,
                                                             type=object_type,
                                                             embeddings=list(),
                                                             locationsDict=dict(),
                                                             matchedBehaviorKeys=set())
                if self.config.clustering.enableOnlineDynamicUpdate and (int(global_id) > self._max_global_id):
                    self._max_global_id = int(global_id)
            if object_type != self._state[global_id].type:
                logging.error(f"ERROR: The object types do not match -- {self._state[global_id].type} != {object_type}.")
                exit(1)
            if global_id in map_global_id_to_mean_embedding.keys():
                self._state[global_id].embeddings = map_global_id_to_mean_embedding[global_id]
            self._state[global_id].locationsDict[timestamp] = locations
            # self._state[global_id].matchedBehaviorKeys.update(matched_behavior_keys)
            self._state[global_id].matchedBehaviorKeys = matched_behavior_keys

            # Convert shadow cluster into normal cluster
            if self.config.clustering.enableOnlineDynamicUpdate:
                if (int(global_id) < 0) and \
                    (timestamp > min(self._state[global_id].locationsDict.keys()) + timedelta(seconds=self.config.clustering.dynamicUpdateLengthThreshSec)):
                    self._max_global_id += 1
                    max_global_id = str(self._max_global_id)
                    self._state[max_global_id] = self._state[global_id]
                    self._state.pop(global_id, None)
                    self._state[max_global_id].id = max_global_id
                    mtmc_objects_plus.locationsOfObjects[i].id = max_global_id
                    for j in range(len(mtmc_objects_plus.objectCounts)):
                        if mtmc_objects_plus.objectCounts[j].type == object_type:
                            mtmc_objects_plus.objectCounts[j].count += 1
                            break
                    global_id = max_global_id

            # Smooth locations
            aggregated_locations: List[List[float]] = list()
            smoothed_locations: List[List[float]] = list()
            timestamp_thresh = max(self._state[global_id].locationsDict.keys()) - timedelta(seconds=self.config.streaming.mtmcPlusSmoothingWindowSec)
            for timestamp, locations in self._state[global_id].locationsDict.items():
                if timestamp > timestamp_thresh:
                    aggregated_locations.extend(locations)
            if len(aggregated_locations) > 0:
                smoothed_locations = [[(sum(coords)/len(coords)) for coords in zip(*aggregated_locations)]]
            mtmc_objects_plus.locationsOfObjects[i].locations = smoothed_locations
            locations_of_objects.append(mtmc_objects_plus.locationsOfObjects[i])

        mtmc_objects_plus.locationsOfObjects = locations_of_objects

        return mtmc_objects_plus

    def delete_older_state(self, deleted_behavior_keys: Set[str]) -> None:
        """
        Deletes global IDs in state when none of the matched behaviors are present in the behavior state

        :param Set[str] deleted_behavior_keys: deleted behavior keys
        :return: None
        ::

            mtmc_plus_state.delete_older_state(deleted_behavior_keys)
        """
        num_locations_to_be_deleted = 0
        global_ids_to_be_deleted: List[str] = list()

        timestamp_current = datetime.utcnow().replace(tzinfo=timezone.utc)
        if self.config.io.inMtmcPlusBatchMode:
            max_timestamp = None
            for global_id in self._state.keys():
                max_location_timestamp = max(list(self._state[global_id].locationsDict.keys()))
                if (max_timestamp is None) or (max_location_timestamp > max_timestamp):
                    max_timestamp = max_location_timestamp
            if max_timestamp is not None:
                timestamp_current = max_timestamp

        for global_id in self._state.keys():
            timestamps_to_be_deleted: List[datetime] = list()

            timestamps = sorted(list(self._state[global_id].locationsDict.keys()))
            for timestamp in timestamps:
                if timestamp_current > timestamp + timedelta(seconds=self.config.preprocessing.mtmcPlusRetentionInStateSec):
                    timestamps_to_be_deleted.append(timestamp)
                    num_locations_to_be_deleted += 1
                    continue
            del timestamps

            for timestamp in timestamps_to_be_deleted:
                self._state[global_id].locationsDict.pop(timestamp, None)
                self._state[global_id].matchedBehaviorKeys = \
                    set([behavior_key for behavior_key in self._state[global_id].matchedBehaviorKeys if behavior_key not in deleted_behavior_keys])

            # Delete global IDs with empty lists of locations
            if len(self._state[global_id].locationsDict) == 0:
                global_ids_to_be_deleted.append(global_id)

        del deleted_behavior_keys

        for global_id in global_ids_to_be_deleted:
            self._state.pop(global_id, None)
            logging.info(f"Removed global ID {global_id} from state")
        del global_ids_to_be_deleted

        if num_locations_to_be_deleted > 0:
            logging.info(f"No. older MTMC location(s) removed from state: {num_locations_to_be_deleted}")

    def clean_unused_state(self, mtmc_objects: List[MTMCObject]) -> None:
        """
        Cleans unused state after stitching global IDs

        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :return: None
        ::

            mtmc_plus_state.clean_unused_state(deleted_behavior_keys)
        """
        used_global_ids: Set[str] = set()
        for mtmc_object in mtmc_objects:
            used_global_ids.add(mtmc_object.globalId)

        global_ids_to_be_deleted: List[str] = list()
        for global_id in self._state.keys():
            if global_id not in used_global_ids:
                global_ids_to_be_deleted.append(global_id)
        del used_global_ids

        for global_id in global_ids_to_be_deleted:
            self._state.pop(global_id, None)
            logging.info(f"Removed global ID {global_id} from state")
        del global_ids_to_be_deleted
