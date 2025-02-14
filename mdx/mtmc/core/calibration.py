# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import logging
import cv2
import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from mdx.mtmc.config import AppConfig
from mdx.mtmc.schema import Sensor, SensorStateObject
from mdx.mtmc.utils.io_utils import validate_file_path, sanitize_string, load_json_from_file


class Calibrator:
    """
    Module to calibrate sensors

    :param Optional[AppConfig] config: configuration for the app
    ::

        calibrator = Calibrator(config)
    """

    def __init__(self, config: Optional[AppConfig]) -> None:
        self.config: Optional[AppConfig] = config

    def _compute_homography(self, image_coordinates: List[Dict[str, int]], global_coordinates: List[Dict[str, int]]) -> Optional[List[List[float]]]:
        """
        Computes the homography matrix from the matched image and global coordinates

        :param List[Dict[str,int]] image_coordinates: list of image coordinates
        :param List[Dict[str,int]] global_coordinates: list of global coordinates
        :return: computed homography or None
        :rtype: Optional[List[List[float]]]
        """
        if len(image_coordinates) != len(global_coordinates):
            logging.warning(f"WARNING: The lengths of image coordinates and global coordinates do NOT match -- "
                            f"{len(image_coordinates)} != {len(global_coordinates)}.")
            return None

        if len(image_coordinates) < 4:
            logging.warning(f"WARNING: The length of coordinates {len(image_coordinates)} is less than 4.")
            return None

        image_coordinate_array: List[List[int]] = list()
        for coord in image_coordinates:
            image_coordinate_array.append((coord["x"], coord["y"]))
        image_coordinate_array = np.array(image_coordinate_array)

        global_coordinate_array: List[List[int]] = list()
        for coord in global_coordinates:
            global_coordinate_array.append((coord["x"], coord["y"]))
        global_coordinate_array = np.array(global_coordinate_array)

        homography, _ = cv2.findHomography(image_coordinate_array, global_coordinate_array,
                                           method=cv2.RANSAC, ransacReprojThreshold=3)
        del image_coordinate_array
        del global_coordinate_array

        return homography.tolist()
    
    def _assume_intrinsic_camera_params(self, attributes: List[Dict[str, str]]) -> Dict[str, List[List[float]]]:
        """
        Assumes intrinsic camera parameters

        :param List[Dict[str,str]] attributes: sensor attributes
        :return: intrinsic camera matrix and distortion coefficients
        :rtype: Dict[str,List[List[float]]]
        """
        dist_coeffs = np.array([[ 0.0, 0.0,  0.0,  0.0,  0.0]])
        
        for attribute in attributes:
            if attribute["name"] == "frameWidth":
                cx = int(attribute["value"]) // 2
            if attribute["name"] == "frameHeight":
                cy = int(attribute["value"]) // 2
                fx = int(attribute["value"])
                fy = int(attribute["value"])

        if (cx is None) or (cy is None):
            logging.error(f"ERROR: The intrinsic camera parameters cannot be computed without frame size given in calibration.")
            exit(1)

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        intrinsic_camera_params = {
            "camera_matrix":camera_matrix.tolist(),
            "dist_coeffs":dist_coeffs.tolist()
        }

        return intrinsic_camera_params
        
    def _compute_extrinsic_camera_params(self, intrinsic_camera_params: Dict[str, List[List[float]]], image_coordinates: List[Dict[str, int]],
                                         global_coordinates: List[Dict[str, int]]) -> Dict[str, List[List[float]]]:
        """
        Computes extrinsic camera parameters

        :param Dict[str,List[List[float]]] intrinsic_camera_params: intrinsic camera matrix and distortion coefficients
        :param List[Dict[str,int]] image_coordinates: list of image coordinates
        :param List[Dict[str,int]] global_coordinates: list of global coordinates
        :return: camera matrix and distortion coefficients
        :rtype: Dict[str,List[List[float]]]
        """
        image_coordinate_array: List[List[int]] = list()
        for coord in image_coordinates:
            image_coordinate_array.append((coord["x"], coord["y"]))
        image_coordinate_array = np.array(image_coordinate_array, dtype=np.float32)

        global_coordinate_array: List[List[int]] = list()
        for coord in global_coordinates:
            global_coordinate_array.append((coord["x"], coord["y"], 0.0))
        global_coordinate_array = np.array(global_coordinate_array, dtype=np.float32)

        camera_matrix = np.array(intrinsic_camera_params["camera_matrix"])
        dist_coeffs = np.array(intrinsic_camera_params["dist_coeffs"])
        flag = 0
        # flag = cv2.SOLVEPNP_IPPE
        success, rotation_vector, translation_vector = cv2.solvePnP(global_coordinate_array, image_coordinate_array, camera_matrix, dist_coeffs, flags=flag)

        if success:
            rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
            camera_position = -np.matrix(rotation_matrix).T * np.matrix(translation_vector)
            roll, pitch, yaw = self._compute_euler_angles_from_rotation_matrix(rotation_matrix)

            camera_params = {
                "camera_position": camera_position.tolist(),
                "rotation_vector": rotation_vector.tolist(),
                "translation_vector": translation_vector.tolist(),
                "rotation_matrix": rotation_matrix.tolist(),
                "euler_angles": [[roll], [pitch], [yaw]]
            }

            return camera_params
        else:
            logging.error(f"ERROR: The extrinsic camera parameters cannot be solved given the image and global coordinates.")
            exit(1)

    def _is_close(self, a: float, b: float, relative_tolerance: float = 1.e-5, absolute_tolerance: float = 1.e-8) -> bool:
        """
        Checks the closeness of two angles

        :param float a: first angle
        :param float b: second angle
        :param float relative_tolerance: relative tolerance
        :param float absolute_tolerance: absolute tolerance
        :return: flag that represents the closeness of two angles
        :rtype: bool
        """
        return abs(a - b) <= absolute_tolerance + relative_tolerance * abs(b)
    
    def _compute_euler_angles_from_rotation_matrix(self, rotation_matrix: np.array) -> Tuple[float, float, float]:
        """
        Computes Euler angles from a rotation matrix

        :param np.array rotation_matrix: rotation matrix
        :return: Euler angles
        :rtype: Tuple[float,float,float]
        """
        phi = 0.0
        if self._is_close(rotation_matrix[2, 0], -1.0):
            theta = math.pi / 2.0
            psi = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
        elif self._is_close(rotation_matrix[2, 0], 1.0):
            theta = -math.pi / 2.0
            psi = math.atan2(-rotation_matrix[0, 1], -rotation_matrix[0, 2])
        else:
            theta = -math.asin(rotation_matrix[2, 0])
            cos_theta = math.cos(theta)
            psi = math.atan2(rotation_matrix[2, 1] / cos_theta, rotation_matrix[2, 2] / cos_theta)
            phi = math.atan2(rotation_matrix[1, 0] / cos_theta, rotation_matrix[0, 0] / cos_theta)
        return psi * 180 / math.pi, theta * 180 / math.pi, phi * 180 / math.pi

    def load_sensors(self, calibration_info: Dict[str, Any]) -> List[Sensor]:
        """
        Loads sensors

        :param Dict[str,Any] calibration_info: calibration information in JSON format
        :return: list of sensors
        :rtype: List[Sensor]
        ::

            sensors = calibrator.load_sensors(calibration_info)
        """
        sensors: List[Sensor] = list()

        for sensor_info in calibration_info["sensors"]:
            translation_to_global_coordinates: Dict[str, float] = {"x": 0.0, "y": 0.0}
            if "translationToGlobalCoordinates" in sensor_info.keys():
                translation_to_global_coordinates = sensor_info["translationToGlobalCoordinates"]
    
            intrinsic_matrix: Optional[List[List[float]]] = None
            if "intrinsicMatrix" in sensor_info.keys():
                intrinsic_matrix = sensor_info["intrinsicMatrix"]

            extrinsic_matrix: Optional[List[List[float]]] = None
            if "extrinsicMatrix" in sensor_info.keys():
                extrinsic_matrix = sensor_info["extrinsicMatrix"]

            camera_matrix: Optional[List[List[float]]] = None
            if "cameraMatrix" in sensor_info.keys():
                camera_matrix = sensor_info["cameraMatrix"]

            homography: Optional[List[List[float]]] = None
            if "homography" in sensor_info.keys():
                homography = sensor_info["homography"]

            tripwires: List[Dict[str, Any]] = list()
            if "tripwires" in sensor_info.keys():
                tripwires = sensor_info["tripwires"]
    
            rois: List[Dict[str, Any]] = list()
            if "rois" in sensor_info.keys():
                rois = sensor_info["rois"]

            sensor = Sensor(type=sensor_info["type"], id=sensor_info["id"], origin=sensor_info["origin"],
                            geoLocation=sensor_info["geoLocation"], coordinates=sensor_info["coordinates"],
                            translationToGlobalCoordinates=translation_to_global_coordinates,
                            scaleFactor=sensor_info["scaleFactor"], attributes=sensor_info["attributes"],
                            place=sensor_info["place"], imageCoordinates=sensor_info["imageCoordinates"],
                            globalCoordinates=sensor_info["globalCoordinates"], intrinsicMatrix=intrinsic_matrix,
                            extrinsicMatrix=extrinsic_matrix, cameraMatrix=camera_matrix, homography=homography,
                            tripwires=tripwires, rois=rois)
            sensors.append(sensor)

        return sensors

    def load_calibration_file(self, calibration_path: str) -> List[Sensor]:
        """
        Loads from a calibration file in JSON format

        :param str calibration_path: path to the calibration file in JSON format
        :return: list of sensors
        :rtype: List[Sensor]
        ::

            sensors = calibrator.load_calibration_file(calibration_path)
        """
        # Load calibration information
        calibration_info = None
        valid_calibration_path = validate_file_path(calibration_path)
        if os.path.exists(valid_calibration_path):
            calibration_info = load_json_from_file(valid_calibration_path)
        else:
            logging.warning(f"WARNING: The calibration file `{valid_calibration_path}` is NOT provided or does NOT exist.")

        # Load sensors
        sensors: List[Sensor] = list()
        if calibration_info is not None:
            sensors = self.load_sensors(calibration_info)

        return sensors

    def convert_sensor_to_sensor_state_object(self, sensor: Sensor) -> SensorStateObject:
        """
        Converts a sensor to a sensor state object

        :param Sensor sensor: sensor object
        :return: converted sensor state object
        :rtype: SensorStateObject
        ::

            sensor_state_object = calibrator.convert_sensor_to_sensor_state_object(sensor)
        """
        logging.info(f"Calibrating sensor {sanitize_string(sensor.id)}...")

        # Read place
        place_str = ""
        place_list = sensor.place
        for i in range(len(place_list)):
            place_segment = f"{place_list[i]['name']}={place_list[i]['value']}"
            if place_str != "":
                place_str += "/"
            place_str += place_segment

        # Read frame size
        frame_width = None
        frame_height = None
        fps = None
        direction = None
        fov_polygon = None
        for attribute in sensor.attributes:
            if attribute["name"] == "frameWidth":
                try:
                    frame_width = int(attribute["value"])
                except ValueError:
                    pass
            if attribute["name"] == "frameHeight":
                try:
                    frame_height = int(attribute["value"])
                except ValueError:
                    pass
            if attribute["name"] == "fps":
                try:
                    fps = float(attribute["value"])
                except ValueError:
                    pass
            if attribute["name"] == "direction":
                try:
                    direction = float(attribute["value"])
                except ValueError:
                    pass
            if attribute["name"] == "fieldOfViewPolygon":
                try:
                    fov_polygon = attribute["value"]
                except ValueError:
                    pass

        # Compute homography
        homography = sensor.homography
        if homography is None:
            homography = self._compute_homography(sensor.imageCoordinates, sensor.globalCoordinates)
        else:
            homography = np.linalg.inv(np.array(homography)).tolist()

        # Create regions of interest
        rois: List[List[Tuple[float, float]]] = list()
        for sensor_roi in sensor.rois:
            roi: List[Tuple[float, float]] = list()
            for coord in sensor_roi["roiCoordinates"]:
                roi.append((coord["x"], coord["y"]))
            rois.append(roi)

        # Create sensor state object
        sensor_state_object = SensorStateObject(placeStr=place_str,
                                                frameWidth=frame_width,
                                                frameHeight=frame_height,
                                                fps=fps,
                                                direction=direction,
                                                fieldOfViewPolygon=fov_polygon,
                                                homography=homography,
                                                rois=rois,
                                                timestamp=None,
                                                sensor=sensor)
        
        if (self.config is not None) and (self.config.localization.rectifyBboxByCalibration):
            # Compute intrinsic camera parameters
            intrinsic_camera_params = self._assume_intrinsic_camera_params(sensor.attributes)
            # Compute extrinsic camera parameters
            extrinsic_camera_params = self._compute_extrinsic_camera_params(intrinsic_camera_params, sensor.imageCoordinates, sensor.globalCoordinates, )
            # Create sensor state object
            sensor_state_object = SensorStateObject(placeStr=place_str,
                                                    frameWidth=frame_width,
                                                    frameHeight=frame_height,
                                                    fps=fps,
                                                    direction=direction,
                                                    fieldOfViewPolygon=fov_polygon,
                                                    homography=homography,
                                                    cameraMatrix=intrinsic_camera_params["camera_matrix"],
                                                    distortionCoeffs=intrinsic_camera_params["dist_coeffs"],
                                                    cameraPosition=extrinsic_camera_params["camera_position"],
                                                    rotationVector=extrinsic_camera_params["rotation_vector"],
                                                    translationVector=extrinsic_camera_params["translation_vector"],
                                                    rotationMatrix=extrinsic_camera_params["rotation_matrix"],
                                                    eulerAngles=extrinsic_camera_params["euler_angles"],
                                                    rois=rois,
                                                    timestamp=None,
                                                    sensor=sensor)

        return sensor_state_object

    def calibrate(self, calibration_path: str) -> Dict[str, SensorStateObject]:
        """
        Calibrates sensors

        :param str calibration_path: path to the calibration file in JSON format
        :return: map from sensor IDs to sensor state objects
        :rtype: Dict[str,SensorStateObject]
        ::

            sensor_state_objects = calibrator.calibrate(calibration_path)
        """
        # Load from a calibration file in JSON format
        sensors = self.load_calibration_file(calibration_path)

        # Convert sensors to sensor state objects
        sensor_state_objects: Dict[str, SensorStateObject] = dict()
        for sensor in sensors:
            sensor_state_object = self.convert_sensor_to_sensor_state_object(sensor)
            sensor_state_objects[sensor.id] = sensor_state_object
        logging.info(f"No. calibrated sensors: {len(sensor_state_objects)}")

        return sensor_state_objects
