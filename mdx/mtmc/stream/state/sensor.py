# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime

from mdx.mtmc.core.calibration import Calibrator
from mdx.mtmc.schema import SensorStateObject, Notification


class StateManager:
    """
    Module to manage sensor state

    ::

        sensor_state = StateManager()
    """

    def __init__(self) -> None:
        self._state: Dict[str, SensorStateObject] = dict()
        self._sensors_deletion_timestamps: Dict[str, Optional[datetime]] = dict()

    def init_state(self, calibrator: Calibrator, calibration_path: str) -> None:
        """
        Initializes state with existing calibration info

        :param Calibrator calibrator: calibrator used to create sensor state objects
        :param str calibration_path: path to the calibration file in JSON format
        :return: None
        ::

            sensor_state.init_state(calibrator, calibration_path)
        """
        self._state = calibrator.calibrate(calibration_path)

    def update_state(self, notifications: List[Notification], calibrator: Calibrator) -> Set[str]:
        """
        Updates sensor state

        :param List[Notification] notifications: list of notifications
        :param Calibrator calibrator: calibrator used to convert sensors to sensor state objects
        :return: updated sensor IDs
        :rtype: Set[str]
        ::

            updated_sensor_ids = sensor_state.update_state(notifications, calibrator)
        """
        updated_sensor_ids: Set[str] = set()

        # Get the last notification with event type of upsert-all
        timestamp_last_upsert_all = None
        for notification in reversed(notifications):
            if notification.event_type == "upsert-all":
                timestamp_last_upsert_all = notification.timestamp
                break

        # Apply the notifications
        for notification in notifications:
            if (timestamp_last_upsert_all is None) or (notification.timestamp >= timestamp_last_upsert_all):
                for sensor in notification.sensors:
                    sensor_id = sensor.id
                    if (sensor_id not in self._state.keys()) or (self._state[sensor_id].timestamp is None) or \
                        (self._state[sensor_id].timestamp < notification.timestamp):

                        # Upsert the sensor state object
                        if notification.event_type in {"upsert-all", "upsert"}:
                            if (sensor_id not in self._sensors_deletion_timestamps.keys()) or \
                                (self._sensors_deletion_timestamps[sensor_id] < notification.timestamp):
                                sensor_state_object = calibrator.convert_sensor_to_sensor_state_object(sensor)
                                sensor_state_object.timestamp = notification.timestamp
                                self._state[sensor_id] = sensor_state_object
                                self._sensors_deletion_timestamps.pop(sensor_id, None)
                                logging.info(f"Upserted sensor {sensor_id}")
                                updated_sensor_ids.add(sensor_id)

                        # Delete the sensor state object
                        elif notification.event_type == "delete":
                            self._state.pop(sensor_id, None)
                            self._sensors_deletion_timestamps[sensor_id] = notification.timestamp
                            logging.info(f"Deleted sensor {sensor_id}")
                            updated_sensor_ids.add(sensor_id)

        return updated_sensor_ids

    def get_sensor_state_objects(self) -> Dict[str, SensorStateObject]:
        """
        Gets a map from all the sensor IDs to sensor state objects in state

        :return: map from sensor IDs to sensor state objects
        :rtype: Dict[str,SensorStateObject]
        ::

            sensor_state_objects = sensor_state.get_sensor_state_objects()
        """
        return self._state
