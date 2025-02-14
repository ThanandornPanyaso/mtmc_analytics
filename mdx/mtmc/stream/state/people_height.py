# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from mdx.mtmc.config import AppConfig
from mdx.mtmc.schema import Bbox, Frame, Behavior, SensorStateObject


class StateManager:
    """
    Module to estimate people's height

    :param AppConfig config: configuration for the app
    ::

        people_height_state = StateManager(config)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self.sensor_state_objects: Optional[Dict[str, SensorStateObject]] = None

        self.map_sensor_id_to_start_time: Dict[str, datetime] = dict()
        self.map_sensor_id_to_current_time: Dict[str, datetime] = dict()
        self.map_sensor_id_to_people_heights: Dict[str, List[float]] = defaultdict(list)
        self.map_sensor_id_to_estimated_people_height: Dict[str, float] = dict()
        self.map_sensor_id_to_is_estimator_ready: Dict[str, bool] = defaultdict(bool)

        if self.config.localization.rectifyBboxByCalibration:
            logging.info(f"Calibration-based bbox rectification is enabled.")
            logging.info("Start estimation of people's height...")
            logging.info(f"Max no. detections for estimation: {self.config.localization.peopleHeightNumSamplesMax}")
            logging.info(f"Max time for estimation: {self.config.localization.peopleHeightMaxLengthSec} seconds")
        else:
            logging.info(f"Calibration-based bbox rectification is disabled.")

    def set_sensor_state_objects(self, sensor_state_objects: Dict[str, SensorStateObject]) -> None:
        """
        Sets sensor state objects

        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :return: None
        ::

            people_height_state.set_sensor_state_objects(sensor_state_objects)
        """
        self.sensor_state_objects = sensor_state_objects

    def estimate_people_height(self, frames: List[Frame]) -> None:
        """
        Estimates people's height from frames

        :param List[Frame] frames: list of frames
        :return: None
        ::

            people_height_state.estimate_people_height(frames)
        """
        if self.config.localization.rectifyBboxByCalibration:
            for frame in frames:
                # Check if the current frame is within the given number of batch frames
                sensor_id = frame.sensorId

                if not self.map_sensor_id_to_is_estimator_ready[sensor_id]:
                    frame_id = int(frame.id)
                    if frame_id <= self.config.localization.peopleHeightNumBatchFrames:
                        for object_instance in frame.objects:
                            # Project the foot point to the ground plane using the homography
                            bbox = object_instance.bbox
                            people_height = self._compute_people_height(bbox, sensor_id)
                            if sensor_id in self.sensor_state_objects.keys():
                                self.map_sensor_id_to_people_heights[sensor_id].append(people_height)

            for sensor_id in self.map_sensor_id_to_people_heights.keys():
                if not self.map_sensor_id_to_is_estimator_ready[sensor_id]:
                    people_heights = self.map_sensor_id_to_people_heights[sensor_id]
                    people_heights.sort(reverse=True)
                    num_estimation_samples = int(len(people_heights) * self.config.localization.peopleHeightEstimationRatio)
                    if self.config.localization.overwrittenPeopleHeightMeter is not None:
                        self.map_sensor_id_to_estimated_people_height[sensor_id] = self.config.localization.overwrittenPeopleHeightMeter
                    else:
                        self.map_sensor_id_to_estimated_people_height[sensor_id] = np.mean(people_heights[:num_estimation_samples])
                    del people_heights
                    self.map_sensor_id_to_is_estimator_ready[sensor_id] = True
                    logging.info(f"Estimated people's height for {sensor_id}: {self.map_sensor_id_to_estimated_people_height[sensor_id]} meters")

    def update_people_height(self, behaviors: List[Behavior]) -> None:
        """
        Updates people's height

        :param List[Behavior] behaviors: list of behaviors
        :return: None
        ::

            people_height_state.update_people_height(behaviors)
        """
        if self.config.localization.rectifyBboxByCalibration:
            for behavior in behaviors:
                sensor_id = behavior.sensorId

                if not self.map_sensor_id_to_is_estimator_ready[sensor_id]:
                    start_time = behavior.timestamp
                    end_time = behavior.end
                    bboxes = behavior.bboxes

                    if sensor_id in self.sensor_state_objects.keys():
                        if sensor_id not in self.map_sensor_id_to_start_time.keys():
                            self.map_sensor_id_to_start_time[sensor_id] = start_time
                            self.map_sensor_id_to_current_time[sensor_id] = end_time
                        if start_time < self.map_sensor_id_to_start_time[sensor_id]:
                            self.map_sensor_id_to_start_time[sensor_id] = start_time
                        if end_time > self.map_sensor_id_to_current_time[sensor_id]:
                            self.map_sensor_id_to_current_time[sensor_id] = end_time

                    for bbox in bboxes:
                        people_height = self._compute_people_height(bbox, sensor_id)
                        if sensor_id in self.sensor_state_objects.keys():
                            self.map_sensor_id_to_people_heights[sensor_id].append(people_height)

                    logging.info(f"Estimation status of {sensor_id}")
                    logging.info(f"No. detections: {len(self.map_sensor_id_to_people_heights[sensor_id])}")
                    if (self.map_sensor_id_to_current_time[sensor_id] > (self.map_sensor_id_to_start_time[sensor_id] + timedelta(seconds=self.config.localization.peopleHeightMaxLengthSec))) or \
                            (len(self.map_sensor_id_to_people_heights[sensor_id]) > self.config.localization.peopleHeightNumSamplesMax):            
                        if len(self.map_sensor_id_to_people_heights[sensor_id]) > 0:
                            people_heights = self.map_sensor_id_to_people_heights[sensor_id]
                            people_heights.sort(reverse=True)
                            num_estimation_samples = int(len(people_heights) * self.config.localization.peopleHeightEstimationRatio)
                            if self.config.localization.overwrittenPeopleHeightMeter is not None:
                                self.map_sensor_id_to_estimated_people_height[sensor_id] = self.config.localization.overwrittenPeopleHeightMeter
                            else:
                                self.map_sensor_id_to_estimated_people_height[sensor_id] = np.mean(people_heights[:num_estimation_samples])
                            del people_heights
                            self.map_sensor_id_to_is_estimator_ready[sensor_id] = True
                            logging.info(f"Estimated people's height: {self.map_sensor_id_to_estimated_people_height[sensor_id]} meters")

    def rectify_bbox(self, foot_pixel: List[float], bbox: Bbox, sensor_id: str) -> Tuple[List[float], float]:
        """
        Computes expected foot location

        :param List[float] foot_pixel: foot pixel location
        :param Bbox bbox: bounding box
        :param str sensor_id: sensor ID
        :return: people's height
        :rtype: float
        ::

            foot_pixel, visibility = people_height_state.rectify_bbox(foot_pixel, bbox, sensor_id)
        """
        if self.map_sensor_id_to_is_estimator_ready[sensor_id]:
            people_height = self.map_sensor_id_to_estimated_people_height[sensor_id]
            foot_pixel_updated, visibility = self._compute_expected_foot_location(bbox, sensor_id, people_height)

            if visibility < self.config.localization.peopleHeightVisibilityThresh:
                foot_pixel_to_return = [foot_pixel_updated[0], foot_pixel_updated[1]]
                del foot_pixel_updated
            else:
                foot_pixel_to_return = foot_pixel
        else:
            foot_pixel_to_return = foot_pixel
            visibility = 1.0
        return foot_pixel_to_return, visibility

    def _compute_expected_foot_location(self, bbox: Bbox, sensor_id: str, people_height: float) -> Tuple[List[float], float]:
        """
        Computes expected foot location

        :param Bbox bbox: bounding box
        :param str sensor_id: sensor ID
        :param float people_height: people height
        :return: foot pixel location and visibility
        :rtype: Tuple[List[float], float]
        """
        # Get sensor state object
        sensor_state_object = self.sensor_state_objects[sensor_id]

        # Get camera parameters
        camera_matrix = np.array(sensor_state_object.cameraMatrix)
        rotation_matrix = np.array(sensor_state_object.rotationMatrix)
        rotation_vector = np.array(sensor_state_object.rotationVector)
        translation_vector = np.array(sensor_state_object.translationVector)
        dist_coeffs = np.array(sensor_state_object.distortionCoeffs)
        del sensor_state_object

        # Get bounding box
        x_min = bbox.leftX
        y_min = bbox.topY
        x_max = bbox.rightX
        y_max = bbox.bottomY

        # Estimate bbox's height
        head_pixel = np.array([[0.5 * (x_min + x_max)], [y_min]], dtype=np.float32)
        scale = self._compute_scale_from_depth(rotation_matrix, translation_vector, camera_matrix, head_pixel, axis="z", depth=people_height)
        head_coord = self._project_image_pixel_to_global_coord(rotation_matrix, translation_vector, camera_matrix, head_pixel, scale=scale)
        del rotation_matrix
        del head_pixel
        foot_coord = head_coord
        foot_coord[2][0] = 0
        image_pixel, _ = cv2.projectPoints(foot_coord, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        del head_coord
        del foot_coord
        del rotation_vector
        del translation_vector
        del camera_matrix
        del dist_coeffs
        foot_pixel = image_pixel[0].T
        y_max_expected = foot_pixel[1][0]
        bbox_height_expected = y_max_expected - y_min
        bbox_height = y_max - y_min

        foot_pixel = [foot_pixel[0][0], foot_pixel[1][0]]
        visibility = min(1.0, bbox_height / bbox_height_expected)

        return foot_pixel, visibility

    def _compute_scale_from_depth(self, rotation_matrix: List[List[float]], translation_vector: List[List[float]], camera_matrix: List[List[float]],
                                  image_pixel: np.array, axis: str, depth: float) -> float:
        """
        Computes scaling factor from depth

        :param List[List[float]] rotation_matrix: rotation matrix
        :param List[List[float]] translation_vector: translation vector
        :param List[List[float]] camera_matrix: camera matrix
        :param np.array image_pixel: image pixel
        :param str axis: axis
        :param float depth: depth for the given axis
        :return: scaling factor
        :rtype: float
        """
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        rotation_matrix_inv = np.linalg.inv(rotation_matrix)
        image_pixel_matrix = np.vstack([image_pixel, 1])
        left = np.matmul(camera_matrix_inv, image_pixel_matrix)
        del camera_matrix_inv
        del image_pixel_matrix
        left = np.matmul(rotation_matrix_inv, left)
        right = np.matmul(rotation_matrix_inv, translation_vector)
        del rotation_matrix_inv
        if axis == "x":
            scale = (depth + right[0][0]) / left[0][0]
        elif axis == "y":
            scale = (depth + right[1][0]) / left[1][0]
        elif axis == "z":
            scale = (depth + right[2][0]) / left[2][0]
        else:
            logging.error(f"ERROR: Undefined axis {axis}.")
            exit(1)
        del left
        del right
        return scale

    def _project_image_pixel_to_global_coord(self, rotation_matrix: List[List[float]], translation_vector: List[List[float]], camera_matrix: List[List[float]],
                                             image_pixel: np.array, scale: float) -> np.array:
        """
        Projects image pixel to global coordinate

        :param List[List[float]] rotation_matrix: rotation matrix
        :param List[List[float]] translation_vector: translation vector
        :param List[List[float]] camera_matrix: camera matrix
        :param np.array image_pixel: image pixel
        :param float scale: scaling factor
        :return: global coordinate
        :rtype: np.array
        """
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        rotation_matrix_inv = np.linalg.inv(rotation_matrix)
        image_pixel_matrix = np.vstack([image_pixel, 1])
        left = np.matmul(camera_matrix_inv, image_pixel_matrix)
        del image_pixel_matrix
        del camera_matrix_inv
        left = np.matmul(rotation_matrix_inv, left)
        left = left * scale
        right = np.matmul(rotation_matrix_inv, translation_vector)
        del rotation_matrix_inv
        return left - right

    def _compute_people_height(self, bbox: Bbox, sensor_id: str) -> float:
        """
        Computes people's height

        :param Bbox bbox: bounding box
        :param str sensor_id: sensor ID
        :return: people's height
        :rtype: float
        """
        # Get sensor state object
        sensor_state_object = self.sensor_state_objects[sensor_id]

        # Get camera parameters
        rotation_matrix = np.array(sensor_state_object.rotationMatrix)
        translation_vector = np.array(sensor_state_object.translationVector)
        camera_matrix = np.array(sensor_state_object.cameraMatrix)
        del sensor_state_object

        # Get bounding box
        x_min = bbox.leftX
        y_min = bbox.topY
        x_max = bbox.rightX
        y_max = bbox.bottomY

        # Estimate people's height
        foot_pixel = np.array([[0.5 * (x_min + x_max)], [y_max]], dtype=np.float32)
        scale_z = self._compute_scale_from_depth(rotation_matrix, translation_vector, camera_matrix, foot_pixel, axis="z", depth=0)
        foot_coord = self._project_image_pixel_to_global_coord(rotation_matrix, translation_vector, camera_matrix, foot_pixel, scale_z)
        del foot_pixel
        depth_x = foot_coord[0][0]
        depth_y = foot_coord[1][0]
        head_pixel = np.array([[0.5 * (x_min + x_max)], [y_min]], dtype=np.float32)
        scale_x = self._compute_scale_from_depth(rotation_matrix, translation_vector, camera_matrix, head_pixel, axis="x", depth=depth_x)
        scale_y = self._compute_scale_from_depth(rotation_matrix, translation_vector, camera_matrix, head_pixel, axis="y", depth=depth_y)
        head_coord_x = self._project_image_pixel_to_global_coord(rotation_matrix, translation_vector, camera_matrix, head_pixel, scale=scale_x)
        head_coord_y = self._project_image_pixel_to_global_coord(rotation_matrix, translation_vector, camera_matrix, head_pixel, scale=scale_y)
        del head_pixel
        del rotation_matrix
        del translation_vector
        del camera_matrix
        height_x = head_coord_x[2][0]
        height_y = head_coord_y[2][0]
        people_height = 0.5 * (height_x + height_y)

        return people_height
