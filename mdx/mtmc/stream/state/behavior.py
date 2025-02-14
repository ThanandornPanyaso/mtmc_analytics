# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
import cv2
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime, timedelta, timezone

from mdx.mtmc.config import AppConfig
from mdx.mtmc.core.data import Preprocessor
from mdx.mtmc.schema import Behavior, BehaviorStateObjects, SensorStateObject


class StateManager:
    """
    Module to manage behavior state

    :param AppConfig config: configuration for the app
    ::

        behavior_state = StateManager(config)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self._state: Dict[str, BehaviorStateObjects] = dict()

    def update_state(self, behaviors: List[Behavior], preprocessor: Preprocessor) -> None:
        """
        Updates behavior state

        :param List[Behavior] behaviors: list of behaviors
        :param Preprocessor preprocessor: preprocessor used to sum embeddings of a behavior
        :return: None
        ::

            behavior_state.update_state(behaviors, preprocessor)
        """
        for behavior in behaviors:
            # Check whether to update or create behavior state object
            create_behavior_state_object = True
            if behavior.id in self._state:
                max_timestamp = self._state[behavior.id].maxTimestamp
                max_end = self._state[behavior.id].behaviorDict[max_timestamp].end
                if behavior.timestamp < (max_end + timedelta(seconds=self.config.preprocessing.behaviorSplitThreshSec)):
                    create_behavior_state_object = False

            # Create a new behavior state object
            if create_behavior_state_object:
                if behavior.id not in self._state.keys():
                    self._state[behavior.id] = BehaviorStateObjects(maxTimestamp=None, behaviorDict=dict())
                self._state[behavior.id].behaviorDict[behavior.timestamp] = preprocessor.sample_locations(behavior)

                if (self._state[behavior.id].maxTimestamp is None) or (behavior.timestamp > self._state[behavior.id].maxTimestamp):
                    self._state[behavior.id].maxTimestamp = behavior.timestamp

            # Update the latest behavior state object
            else:
                max_timestamp = self._state[behavior.id].maxTimestamp
                if behavior.end > self._state[behavior.id].behaviorDict[max_timestamp].end:
                    self._state[behavior.id].behaviorDict[max_timestamp].end = behavior.end
                if int(behavior.endFrame) > int(self._state[behavior.id].behaviorDict[max_timestamp].endFrame):
                    self._state[behavior.id].behaviorDict[max_timestamp].endFrame = behavior.endFrame
                self._state[behavior.id].behaviorDict[max_timestamp].timestamps += behavior.timestamps
                self._state[behavior.id].behaviorDict[max_timestamp].frameIds += behavior.frameIds
                self._state[behavior.id].behaviorDict[max_timestamp].bboxes += behavior.bboxes
                self._state[behavior.id].behaviorDict[max_timestamp].confidences += behavior.confidences
                self._state[behavior.id].behaviorDict[max_timestamp].locations += behavior.locations
                self._state[behavior.id].behaviorDict[max_timestamp].locationMask += behavior.locationMask

                if len(self._state[behavior.id].behaviorDict[max_timestamp].embeddings) == 0:
                    self._state[behavior.id].behaviorDict[max_timestamp].embeddings = behavior.embeddings
                elif len(behavior.embeddings) > 0:
                    for i in range(len(behavior.embeddings[0])):
                        self._state[behavior.id].behaviorDict[max_timestamp].embeddings[0][i] += behavior.embeddings[0][i]

                self._state[behavior.id].behaviorDict[max_timestamp] = preprocessor.sample_locations(self._state[behavior.id].behaviorDict[max_timestamp])

    def update_places_and_locations(self, sensor_state_objects: Dict[str, SensorStateObject], updated_sensor_ids: Set[str]) -> None:
        """
        Updates places and locations of behavior state objects

        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :param Set[str] updated_sensor_ids: updated sensor IDs
        :return: None
        ::

            behavior_state.update_places_and_locations(sensor_state_objects, updated_sensor_ids)
        """
        for behavior_id in self._state.keys():
            for timestamp in self._state[behavior_id].behaviorDict.keys():
                sensor_id = self._state[behavior_id].behaviorDict[timestamp].sensorId
                if sensor_id in updated_sensor_ids:
                    # Reset locations
                    self._state[behavior_id].behaviorDict[timestamp].locations.clear()

                    # Handle upserted sensor
                    if sensor_id in sensor_state_objects.keys():
                        self._state[behavior_id].behaviorDict[timestamp].place = sensor_state_objects[sensor_id].placeStr
                        if sensor_state_objects[sensor_id].homography is not None:
                            homography = np.array(sensor_state_objects[sensor_id].homography)
                            location_mask = self._state[behavior_id].behaviorDict[timestamp].locationMask
                            bboxes = self._state[behavior_id].behaviorDict[timestamp].bboxes
                            for i in range(len(bboxes)):
                                if not location_mask[i]:
                                    continue
                                foot_pixel = [((bboxes[i].leftX + bboxes[i].rightX) / 2.0), bboxes[i].bottomY]
                                foot_pixel = np.float32([foot_pixel]).reshape(-1, 1, 2)
                                location = cv2.perspectiveTransform(foot_pixel, homography).squeeze(1)
                                del foot_pixel
                                location = [float(location[0, 0]), float(location[0, 1])]
                                self._state[behavior_id].behaviorDict[timestamp].locations.append(location)
                            del homography

    def get_behavior(self, behavior_id: str) -> Optional[Behavior]:
        """
        Gets a behavior given the behavior ID

        :param str behavior_id: behavior ID
        :return: behavior
        :rtype: Optional[Behavior]
        ::

            behavior = behavior_state.get_behavior(behavior_id)
        """
        same_id_behavior_state_objects = self._state.get(behavior_id, None)
        if same_id_behavior_state_objects is None:
            return None
        return same_id_behavior_state_objects.behaviorDict[same_id_behavior_state_objects.maxTimestamp]

    def get_behaviors(self, behavior_ids: List[str]) -> List[Optional[Behavior]]:
        """
        Gets a list of behaviors given the behavior IDs

        :param List[str] behavior_ids: list of behavior IDs
        :return: list of behaviors
        :rtype: List[Optional[Behavior]]
        ::

            behaviors = behavior_state.get_behaviors(behavior_ids)
        """
        return [self.get_behavior(behavior_id) for behavior_id in behavior_ids]

    def get_behaviors_in_state(self) -> List[Behavior]:
        """
        Gets all the behaviors in state

        :return: list of behaviors
        :rtype: List[Behavior]
        ::

            behaviors = behavior_state.get_behaviors_in_state()
        """
        behaviors: List[Behavior] = list()
        for behavior_id in self._state.keys():
            for timestamp in self._state[behavior_id].behaviorDict.keys():
                behavior = self._state[behavior_id].behaviorDict[timestamp].copy(deep=False)
                behavior.timestamps = self._state[behavior_id].behaviorDict[timestamp].timestamps.copy()
                behavior.locations = self._state[behavior_id].behaviorDict[timestamp].locations.copy()
                behavior.locationMask = self._state[behavior_id].behaviorDict[timestamp].locationMask.copy()
                behavior.embeddings = self._state[behavior_id].behaviorDict[timestamp].embeddings.copy()
                behaviors.append(behavior)
        return behaviors

    def get_behavior_ids_in_state(self) -> Set[str]:
        """
        Gets all the behavior IDs in state

        :return: set of behavior IDs
        :rtype: Set[str]
        ::

            behavior_ids = behavior_state.get_behavior_ids_in_state()
        """
        return set(self._state.keys())

    def get_behavior_keys_in_state(self) -> Set[str]:
        """
        Gets all the behavior keys (sensor-object-timestamp IDs) in state

        :return: set of behavior keys (sensor-object-timestamp IDs)
        :rtype: Set[str]
        ::

            behavior_keys = behavior_state.get_behavior_keys_in_state()
        """
        behavior_keys: Set[str] = set()
        for behavior_id in self._state.keys():
            for timestamp in self._state[behavior_id].behaviorDict.keys():
                behavior_keys.add(behavior_id + " #-# " + timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"))
                
        return behavior_keys

    def delete_older_state(self) -> Set[str]:
        """
        Deletes behaviors in state that have not been updated for a configurable time interval

        :return: deleted behavior keys
        :return: Set[str]
        ::

            deleted_behavior_keys = behavior_state.delete_older_state()
        """
        behaviors_to_be_deleted: Set[Tuple[str, int]] = set()

        timestamp_current = datetime.utcnow().replace(tzinfo=timezone.utc)
        if self.config.io.inMtmcPlusBatchMode:
            max_behavior_end = None
            for behavior_id in self._state.keys():
                for timestamp in self._state[behavior_id].behaviorDict.keys():
                    behavior_end = self._state[behavior_id].behaviorDict[timestamp].end
                    if (max_behavior_end is None) or (behavior_end > max_behavior_end):
                        max_behavior_end = behavior_end
            if max_behavior_end is not None:
                timestamp_current = max_behavior_end

        for behavior_id in self._state.keys():
            for timestamp in self._state[behavior_id].behaviorDict.keys():
                behavior_end = self._state[behavior_id].behaviorDict[timestamp].end
                if timestamp_current > behavior_end + timedelta(seconds=self.config.preprocessing.behaviorRetentionInStateSec):
                    behaviors_to_be_deleted.add((behavior_id, timestamp))

        for behavior_id in list(self._state):
            for timestamp in list(self._state[behavior_id].behaviorDict):
                # Delete the behavior state object
                if (behavior_id, timestamp) in behaviors_to_be_deleted:
                    self._state[behavior_id].behaviorDict.pop(timestamp, None)
                    # Delete the behavior ID if there is no alive behavior state object
                    if len(self._state[behavior_id].behaviorDict) == 0:
                        self._state.pop(behavior_id, None)
                        break
                    # Reset maximum timestamp
                    if timestamp == self._state[behavior_id].maxTimestamp:
                        max_timestamp = None
                        for timestamp in self._state[behavior_id].behaviorDict.keys():
                            if (max_timestamp is None) or (timestamp > max_timestamp):
                                max_timestamp = timestamp
                        self._state[behavior_id].maxTimestamp = max_timestamp

        if len(behaviors_to_be_deleted) > 0:
            logging.info(f"No. older behaviors removed from state: {len(behaviors_to_be_deleted)}")
            logging.info(f"No. remaining behaviors in state: {len(self._state.keys())}")

        return set([behavior_id + " #-# " + timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                    for (behavior_id, timestamp) in behaviors_to_be_deleted])
