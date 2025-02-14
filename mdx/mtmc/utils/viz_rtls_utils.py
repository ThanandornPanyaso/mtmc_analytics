# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import logging
import cv2
import json
import bisect
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from shapely import wkt

from mdx.mtmc.schema import SensorStateObject
from mdx.mtmc.config import VizRtlsConfig
from mdx.mtmc.core.calibration import Calibrator

logging.basicConfig(format="%(asctime)s.%(msecs)03d - %(message)s",
                    datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)


class VizConfig:
    """
    Visualization config

    :param VizRtlsConfig config: configuration for RTLS visualization
    ::

        viz_config = VizConfig(config)
    """

    def __init__(self, config: VizRtlsConfig) -> None:
        self.rtls_config: VizRtlsConfig = config
        image_map = cv2.imread(config.input.mapPath)
        self.map_height = image_map.shape[0]
        self.map_width = image_map.shape[1]
        self.map_aspect_ratio = self.map_width / self.map_height
        calibrator = Calibrator(None)
        self.sensor_state_objects = calibrator.calibrate(config.input.calibrationPath)
        first_sensor_id = list(self.sensor_state_objects.keys())[0]
        self.translation_to_global_coordinates = self.sensor_state_objects[first_sensor_id].sensor.translationToGlobalCoordinates
        self.scale_factor = self.sensor_state_objects[first_sensor_id].sensor.scaleFactor
        # Assume all sensors share the same FPS and frame size
        self.fps = self.sensor_state_objects[first_sensor_id].fps
        self.frame_width = self.sensor_state_objects[first_sensor_id].frameWidth
        self.frame_height = self.sensor_state_objects[first_sensor_id].frameHeight
        self.output_map_height = self.rtls_config.output.outputMapHeight
        self.output_map_width = int(self.output_map_height * self.map_aspect_ratio)
        self.output_video_size = (self.output_map_width, self.output_map_height)
        self.sensor_views_layout = config.output.sensorViewsLayout
        self.activated_sensor_ids: List[str] = list()

        if config.output.displaySensorViews:
            # This needs to match the output video size (W, H), which is specific to the sensor setup
            map_to_frame_height_ratio = float(self.output_map_height) / float(self.frame_height)
            map_to_frame_width_ratio = float(self.output_map_width) / float(self.frame_width)
            map_height_40 = self.output_map_height + (2 * int(self.frame_height * map_to_frame_width_ratio / 5))
            map_to_frame_height_ratio_40 = float(map_height_40) / float(self.frame_height)
            map_height_100 = self.output_map_height + (4 * int(self.frame_height * map_to_frame_width_ratio / 8))
            map_to_frame_height_ratio_100 = float(map_height_100) / float(self.frame_height)

            map_num_sensors_to_radial_output_size: Dict[int, Tuple[int, int]] = {  
                8: (self.output_map_width + (2 * int(self.frame_width * map_to_frame_height_ratio / 4)), self.output_map_height),
                12: (self.output_map_width + (4 * int(self.frame_width * map_to_frame_height_ratio / 3)), self.output_map_height),
                16: (self.output_map_width + (4 * int(self.frame_width * map_to_frame_height_ratio / 4)), self.output_map_height),
                30: (self.output_map_width + (6 * int(self.frame_width * map_to_frame_height_ratio / 5)), self.output_map_height),
                40: (self.output_map_width + (6 * int(self.frame_width * map_to_frame_height_ratio_40 / 7)), map_height_40),
                96: (self.output_map_width + (6 * int(self.frame_width * map_to_frame_height_ratio_100 / 12)), map_height_100),
                100: (self.output_map_width + (6 * int(self.frame_width * map_to_frame_height_ratio_100 / 12)), map_height_100)
            }

            map_num_sensors_to_split_output_size: Dict[int, Tuple[int, int]] = {  
                8: (self.output_map_width + (2 * int(self.frame_width * map_to_frame_height_ratio / 4)), self.output_map_height),
                12: (self.output_map_width + (3 * int(self.frame_width * map_to_frame_height_ratio / 4)), self.output_map_height),
                16: (self.output_map_width + (4 * int(self.frame_width * map_to_frame_height_ratio / 4)), self.output_map_height),
                30: (self.output_map_width + (5 * int(self.frame_width * map_to_frame_height_ratio / 6)), self.output_map_height),
                40: (self.output_map_width + (5 * int(self.frame_width * map_to_frame_height_ratio / 8)), self.output_map_height),
                96: (self.output_map_width + (8 * int(self.frame_width * map_to_frame_height_ratio / 12)), self.output_map_height),
                100: (self.output_map_width + (10 * int(self.frame_width * map_to_frame_height_ratio / 10)), self.output_map_height)
            }

            if self.sensor_views_layout == "radial":
                self.output_video_size = map_num_sensors_to_radial_output_size[config.output.sensorSetup.value]
            elif self.sensor_views_layout == "split":
                self.output_video_size = map_num_sensors_to_split_output_size[config.output.sensorSetup.value]
            else:
                logging.error(f"ERROR: The sensor views' layout {self.sensor_views_layout} is not defined.")
                exit(1)


class GlobalObject:
    """
    Global object with locations and corresponding timestamps

    :param VizConfig config: visualization config
    :param int global_id: global ID
    :param Optional[Tuple[int,int,int]] color: color
    ::

        global_object = GlobalObject(global_id, color)
    """

    def __init__(self, config: VizConfig, global_id: int, color: Optional[Tuple[int, int, int]] = None):
        self.global_id = global_id
        if color is not None:
            self.color = color
        else:
            color = np.random.random(3) * 255
            self.color = color.tolist()
        self.locations: List[List[float]] = list()
        self.timestamps: List[str] = list()
        self.buffered_locations: List[List[float]] = list()
        self.buffer_length_thresh_sec = config.rtls_config.output.bufferLengthThreshSec
        self.trajectory_length_thresh_sec = config.rtls_config.output.trajectoryLengthThreshSec
        self.buffered_timestamps: List[str] = list()
        self.length = 0
        self.is_active = False
        self.buffer_length_thresh_timedelta = timedelta(milliseconds=1000*self.buffer_length_thresh_sec)
        self.trajectory_length_thresh_timedelta = timedelta(milliseconds=1000*self.trajectory_length_thresh_sec)

    def update(self, location: List[float], timestamp: str, enable_smoothing: bool = False) -> None:
        """
        Updates locations and timestamps of the object

        :param List[float] location: location
        :param str timestamp: timestamp
        :param bool enable_smoothing: flag indicating whether to apply smoothing
        :return: None
        ::

            global_object.update(location, timestamp, enable_smoothing)
        """
        self.buffered_locations.append(location)
        self.buffered_timestamps.append(timestamp.split("Z")[0])
        self.buffered_locations, self.buffered_timestamps = self._trim_locations_based_on_timestamps(self.buffered_locations, self.buffered_timestamps, self.buffer_length_thresh_timedelta)

        if enable_smoothing:
            self.locations.append(np.mean(np.array(self.buffered_locations), axis=0).tolist())
            self.timestamps.append("")
        else:
            self.locations.append(location)
            self.timestamps.append(timestamp.split("Z")[0])

        self.locations, self.timestamps = self._trim_locations_based_on_timestamps(self.locations, self.timestamps, self.trajectory_length_thresh_timedelta)
        self.length = len(self.locations)

    def _trim_locations_based_on_timestamps(self, locations: List[List[float]], timestamps: List[str], time_span_thresh_timedelta: datetime) -> Tuple[List[List[float]], List[str]]:
        """
        Trims locations based on timestamps

        :param List[List[float]] locations: locations
        :param List[str] timestamps: timestamps
        :param datetime time_span_thresh_timedelta: threshold of time span
        :return: trimmed locations and timestamps
        :rtype: Tuple[List[List[float]],List[str]]
        """
        if len(timestamps) > 1:
            time_span = datetime.fromisoformat(timestamps[-1]) - datetime.fromisoformat(timestamps[0])
            while time_span > time_span_thresh_timedelta:
                locations.pop(0)
                timestamps.pop(0)
                if len(timestamps) == 1:
                    break
                time_span = datetime.fromisoformat(timestamps[-1]) - datetime.fromisoformat(timestamps[0])
        return locations, timestamps

    def activate(self) -> None:
        """
        Activates the global object

        :return: None
        ::

            global_object.activate()
        """
        self.is_active = True

    def deactivate(self) -> None:
        """
        Deactivates the global object

        :return: None
        ::

            global_object.deactivate()
        """
        self.is_active = False


class GlobalObjects:
    """
    Global objects

    :param VizConfig config: visualization config
    ::

        global_objects = GlobalObjects(config)
    """

    def __init__(self, config: VizConfig) -> None:
        self.config = config
        self.translation_to_global_coordinates = config.translation_to_global_coordinates
        self.scale_factor = config.scale_factor
        self.map_height = config.map_height
        self.trajectory_length_thresh = int(config.rtls_config.output.trajectoryLengthThreshSec * config.fps)
        self.current_frame_id = -1
        self.color_assignments: List[str] = list()
        self.global_objects: Dict[int, GlobalObject] = dict()
        self.colors = [
            (255, 26, 159), # ff1a9f 
            (146, 218, 98), # 92da62 
            (171, 148, 255), # ab94ff 
            (202, 181, 59), # cab53b 
            (153, 69, 6), # 994506 
            (165, 220, 150), # a5dc96 
            (89, 64, 181), # 5940b5 
            (164, 0, 166), # a400a6 
            (109, 253, 219), # 6dfddb 
            (247, 133, 43), # f7852b 
            (13, 37, 135), # 0d2587 
            (251, 227, 121), # fbe379 
            (255, 107, 15), # ff6b0f 
            (0, 149, 194), # 0095c2 
            (126, 116, 73), # 7e7449 
            (230, 52, 76), # e6344c 
            (255, 148, 210), # ff94d2 
            (139, 255, 61), # 8bff3d 
            (248, 225, 93), # f8e15d 
            (3, 196, 151), # 03c497 
            (53, 0, 245), # 3500f5 
            (215, 204, 255), # d7ccff 
            (252, 77, 255), # fc4dff 
            (250, 178, 122), # fab27a 
            (59, 95, 237), # 3b5fed 
            (205, 198, 168), # cdc6a8 
            (244, 169, 179), # f4a9b3 
            (255, 176, 128), # ffb080 
            (112, 222, 255), # 70deff 
            (223, 255, 0), # DFFF00 
            (152, 171, 245), # 98abf5 
            (254, 184, 255) # feb8ff
        ]
        color_assignments = -1 * np.ones(len(self.colors))
        self.color_assignments = color_assignments.tolist()
        
    def _assign_color(self, global_id: int) -> Tuple[int, int, int]:
        """
        Assigns color for a global ID

        :param int global_id: str
        :return: color
        :rtype: Tuple[int, int, int]
        """
        min_assigned_global_id = min(self.color_assignments)
        assigned_color_idx = self.color_assignments.index(min_assigned_global_id)
        assigned_color = self.colors[assigned_color_idx]
        self.color_assignments[assigned_color_idx] = global_id
        return assigned_color

    def update(self, locations_of_objects: Dict[str, Any], frame_id: int, timestamp: str) -> None:
        """
        Updates global objects

        :param Dict[str, Any] locations_of_objects: locations of objects
        :param int frame_id: frame ID
        :param str timestamp: timestamp
        :return: None
        ::

            global_objects.update(locations_of_objects, frame_id, timestamp)
        """
        active_global_ids: List[int] = list()

        if frame_id > self.current_frame_id:
            self.current_frame_id = frame_id

            for locations_of_object in locations_of_objects:
                global_id = int(locations_of_object["id"])
                if len(locations_of_object["locations"]) > 0:
                    location = locations_of_object["locations"][0]
                    if global_id not in self.global_objects.keys():
                        assigned_color = self._assign_color(global_id)
                        self.global_objects[global_id] = GlobalObject(self.config, global_id, assigned_color)
                    self.global_objects[global_id].update(location,timestamp)
                    self.global_objects[global_id].activate()
                    active_global_ids.append(global_id)

            for global_id in self.global_objects.keys():
                if global_id not in active_global_ids:
                    self.global_objects[global_id].deactivate()

        del active_global_ids

    def get_trajectory_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Gets trajectory information

        :return: trajectory information
        :rtype: Dict[int, Dict[str, Any]]
        ::

            map_global_id_to_trajectory_info = global_objects.get_trajectory_info()
        """
        map_global_id_to_trajectory_info: Dict[int, Dict[str, Any]] = dict()

        for global_id in self.global_objects.keys():
            if self.global_objects[global_id].is_active:
                global_object = self.global_objects[global_id]
                color = global_object.color
                trajectory: List[Tuple[int, int]] = list()
                for i in range(global_object.length):
                    location = global_object.locations[i]
                    location = convert_to_map_pixel(location, self.translation_to_global_coordinates, self.scale_factor, self.map_height)
                    trajectory.append(location)
                    if len(trajectory) > self.trajectory_length_thresh:
                        trajectory.pop(0)
                map_global_id_to_trajectory_info[global_id] = {
                    "color": color,
                    "trajectory": trajectory
                }

        return map_global_id_to_trajectory_info


def read_rtls_log(rtls_log_path: str) -> Tuple[List[str], List[int]]:
    """
    Reads RTLS log file

    :param str rtls_log_path: RTLS log file path
    :return: RTLS log and frame IDs
    :rtype: Tuple[List[str],List[int]]
    ::

        rtls_log, frame_ids = read_rtls_log(rtls_log_path)
    """
    rtls_log: List[str] = list()
    frame_ids: List[int] = list()

    if os.path.exists(rtls_log_path):
        with open(rtls_log_path, "r") as f:
            rtls_log = f.readlines()

        for line in rtls_log:
            line = line.rstrip()
            # Remove the "b" symbol (byte) to read the line correctly 
            if line.startswith("b'"):
                line = line[2:-1]
            rtls_log_line = json.loads(line)
            frame_id = int(rtls_log_line["frameId"])
            frame_ids.append(frame_id)

        logging.info(f"Loaded RTLS log from {rtls_log_path}")
    else:
        logging.error(f"ERROR: The RTLS log {rtls_log_path} does not exist.")
        exit(1)

    return rtls_log, frame_ids


def read_protobuf_data_with_amr_data(protobuf_data_path: str) -> Tuple[Dict[int, Dict[str, Dict[str, Any]]], List[str], List[int]]:
    """
    Reads protobuf data file with AMR data

    :param str protobuf_data_path: protobuf data file path
    :return: dictionary of protobuf data, AMR log, and AMR frame IDs 
    :rtype: Tuple[Dict[int,Dict[str,Dict[str,Any]]],List[str],List[int]]
    ::

        protobuf_data_dict, amr_log, amr_frame_ids = read_protobuf_data_with_amr_data(protobuf_data_path)
    """
    protobuf_data_dict: Dict[int, Dict[str, Dict[str, Any]]] = dict()
    amr_log: List[str] = list()
    amr_frame_ids: List[int] = list()

    if os.path.exists(protobuf_data_path):
        with open(protobuf_data_path,"r") as f:
            protobuf_data = f.readlines()

        num_amr_pending_frame_ids = 0

        for line in protobuf_data:
            protobuf_data_line = json.loads(line)
            if 'locationsOfObjects' in protobuf_data_line.keys():
                num_amr_pending_frame_ids += 1
                amr_log.append(line)
            else:
                frame_id = int(protobuf_data_line["id"])
                while num_amr_pending_frame_ids > 0:
                    amr_frame_ids.append(frame_id)
                    num_amr_pending_frame_ids -= 1

                if frame_id not in protobuf_data_dict:
                    protobuf_data_dict[frame_id] = dict()
                sensor_id = protobuf_data_line["sensorId"]
                objects = protobuf_data_line["objects"]

                map_object_id_to_info: Dict[str, Any] = dict()
                for object_instance in objects:
                    object_id = object_instance["id"]
                    left_x = object_instance["bbox"]["leftX"]
                    top_y = object_instance["bbox"]["topY"]
                    right_x = object_instance["bbox"]["rightX"]
                    bottom_y = object_instance["bbox"]["bottomY"]
                    bbox = (left_x, top_y, right_x, bottom_y)
                    if "footLocation" in object_instance["info"].keys():
                        foot_tokens = object_instance["info"]["footLocation"].split(",")
                        foot_location = [float(foot_tokens[0]), float(foot_tokens[1])]
                    else:
                        foot_location = [(left_x + right_x) / 2., bottom_y]
                    map_object_id_to_info[object_id] = bbox, foot_location
                protobuf_data_dict[frame_id][sensor_id] = map_object_id_to_info

        logging.info(f"Loaded protobuf data from {protobuf_data_path}")

    else:
        logging.info(f"{protobuf_data_path} does not exist")

    return protobuf_data_dict, amr_log, amr_frame_ids


def read_json_data(json_data_path: str) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Reads JSON data file

    :param str json_data_path: JSON data file path
    :return: dictionary of JSON data
    :rtype: Dict[int,Dict[str,Dict[str,Any]]]
    ::

        json_data_dict = read_json_data(json_data_path)
    """
    with open(json_data_path,"r") as f:
        json_data = f.readlines()

    json_data_dict: Dict[int, Dict[str, Dict[str, Any]]] = dict()
    for line in json_data:
        json_data_line = json.loads(line)
        frame_id = int(json_data_line["id"])
        if frame_id not in json_data_dict:
            json_data_dict[frame_id] = dict()
        sensor_id = json_data_line["sensorId"]
        objects = json_data_line["objects"]

        map_object_id_to_info: Dict[str, Any] = dict()
        for object_instance in objects:
            object_tokens = object_instance.split("|")
            object_id = int(object_tokens[0])
            left_x = float(object_tokens[1])
            top_y = float(object_tokens[2])
            right_x = float(object_tokens[3])
            bottom_y = float(object_tokens[4])
            bbox = (left_x, top_y, right_x, bottom_y)
            foot_location = [(left_x + right_x) / 2., bottom_y]
            map_object_id_to_info[object_id] = bbox, foot_location
        json_data_dict[frame_id][sensor_id] = map_object_id_to_info

    logging.info(f"Loaded JSON data from {json_data_path}")

    return json_data_dict


def read_videos(video_dir_path: str) -> Dict[str, cv2.VideoCapture]:
    """
    Reads videos

    :param str video_dir_path: videos directory path
    :return: map from video names to video captures
    :rtype: Dict[str, cv2.VideoCapture]
    ::

        map_video_name_to_capture = read_videos(video_dir_path)
    """
    map_video_name_to_capture: Dict[str, cv2.VideoCapture] = dict()
    video_names = os.listdir(video_dir_path)
    for video_name in video_names:
        video_capture = cv2.VideoCapture(os.path.join(video_dir_path, video_name))
        map_video_name_to_capture[video_name] = video_capture
    del video_names

    logging.info(f"Loaded videos from {video_dir_path}")   

    return map_video_name_to_capture


def read_topview_video(topdown_video_path: str) -> cv2.VideoCapture:
    """
    Reads top-view video

    :param str topdown_video_path: top-view video file path
    :return: top-view video capture
    :rtype: cv2.VideoCapture
    ::

        topview_video_capture = read_topview_video(topdown_video_path)
    """
    return cv2.VideoCapture(topdown_video_path)


def convert_to_map_pixel(location: List[float], translation_to_global_coordinates: Dict[str, float], scale_factor: float, map_height: int) -> Tuple[int, int]:
    """
    Converts global location to pixel location on the map

    :param List[float] location: location
    :param Dict[str, float] translation_to_global_coordinates: translation to global coordinates
    :param float scale_factor: scale factor
    :param int map_height: map height
    :return: map pixel
    :rtype: Tuple[int, int]
    ::

        map_pixel = convert_to_map_pixel_location(location, translation_to_global_coordinates, scale_factor, map_height)
    """
    return int((location[0] + translation_to_global_coordinates["x"]) * scale_factor), \
        int(map_height - 1. - ((location[1] + translation_to_global_coordinates["y"]) * scale_factor))


def pad_image(image: np.array, boundary_width: int = 3, color: Tuple[int, int, int] = (255, 255, 255)) -> np.array:
    """
    Pads an image

    :param np.array image: image
    :param int boundary_width: boundary width
    :param Tuple[int,int,int] color: color
    :return: padded image
    :rtype: np.array
    ::

        padded_image = pad_image(image, boundary_width, color)
    """
    image_copy = np.copy(image)
    r, g, b = color

    image_copy[:, :boundary_width, 0] = r
    image_copy[:, -boundary_width:, 0] = r
    image_copy[:boundary_width, :, 0] = r
    image_copy[-boundary_width:, :, 0] = r

    image_copy[:, :boundary_width, 1] = g
    image_copy[:, -boundary_width:, 1] = g
    image_copy[:boundary_width, :, 1] = g
    image_copy[-boundary_width:, :, 1] = g

    image_copy[:, :boundary_width, 2] = b
    image_copy[:, -boundary_width:, 2] = b
    image_copy[:boundary_width, :, 2] = b
    image_copy[-boundary_width:, :, 2] = b

    return image_copy


def darken_image(image: np.array, alpha: float = 0.2) -> np.array:
    """
    Darkens an image

    :param np.array image: image
    :param float alpha: ratio of original image
    :return: darkened image
    :rtype: np.array
    ::

        darkened_image = darken_image(image, alpha)
    """
    blank_image = np.zeros(image.shape)
    image = np.asarray(image, np.uint8)
    blank_image = np.asarray(blank_image, np.uint8)
    cv2.addWeighted(image, alpha, blank_image, 1. - alpha, 0, image)
    
    return image


def shift_center(center: Tuple[int, int], radius: float, angle: float) -> Tuple[int, int]:
    """
    Shifts center point of a fan

    :param Tuple[int,int] center: center point
    :param float radius: radius of the fan
    :param float angle: angle of the fan in degree
    :return: shifted center point
    :rtype: Tuple[int,int]
    ::

        shifted_center = shift_center(center, radius, angle)
    """
    angle = (90 - angle) / 180 * np.pi
    y_shift = radius * np.sin(angle)
    x_shift = radius * np.cos(angle)
    x, y = center

    return (int(x + x_shift), int(y - y_shift))


def plot_fan_shape(image_map: np.array, location: List[float], start_angle: float, end_angle: float, radius: float, color: Tuple[int, int, int] = (242, 227, 227)) -> np.array:
    """
    Plots fan shape

    :param np.array image_map: image of floor plan
    :param List[float] location: location
    :param float start_angle: starting angle of the fan in degree
    :param float end_angle: ending angle of the fan in degree
    :param float radius: radius of the fan
    :param Tuple[int,int,int] color: color
    :return: plotted image
    :rtype: np.array
    ::

        plotted = plot_fan_shape(image_map, location, start_angle, end_angle, radius, color)
    """
    center = (int(location[0]), int(location[1]))
    edges = [shift_center(center, radius, start_angle),center, shift_center(center, radius, end_angle)]
    image_map = cv2.polylines(image_map, np.int32([np.array(edges).reshape(-1, 1, 2)]), False, color=color, thickness=1)
    del edges
    image_map = cv2.ellipse(image_map, center, axes=(int(radius), int(radius)), angle=270, startAngle=int(start_angle), endAngle=int(end_angle), color=color, thickness=1)
    del center

    return image_map


def plot_sensor_icon(image_map: np.array, sensor_state_object: SensorStateObject, radius: float = 50, half_span: float = 40):
    """
    Plots sensor icon

    :param np.array image_map: image of floor plan
    :param SensorStateObject sensor_state_object: sensor state object
    :param float radius: radius of the fan
    :param float half_span: half span of the fan in degree
    :return: plotted image
    :rtype: np.array
    ::

        plotted_image = plot_sensor_icon(image_map, sensor_state_object, radius, half_span)
    """
    image_copy = np.copy(image_map)

    sensor_id = sensor_state_object.sensor.id
    coordinates = sensor_state_object.sensor.coordinates
    coordinates = convert_to_map_pixel([coordinates["x"], coordinates["y"]], sensor_state_object.sensor.translationToGlobalCoordinates,
                                       sensor_state_object.sensor.scaleFactor, image_map.shape[0])
    start_angle = sensor_state_object.direction - half_span
    end_angle = sensor_state_object.direction + half_span
    image_copy = plot_fan_shape(image_copy, coordinates, start_angle, end_angle, radius, color=(255,255,255))

    text_center = (int(coordinates[0]) + 10, int(coordinates[1]) - 10)

    image_copy = cv2.putText(image_copy,"C{}".format(sensor_id[-2:]), text_center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    image_copy = cv2.putText(image_copy,"C{}".format(sensor_id[-2:]), text_center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    image_copy = cv2.circle(image_copy, (int(coordinates[0]), int(coordinates[1])), 4, (255, 0, 0), thickness=-1)
    image_copy = cv2.circle(image_copy, (int(coordinates[0]), int(coordinates[1])), 3, (255, 255, 255), thickness=-1)

    return image_copy


def correct_fov_polygon(fov_polygon: str) -> str:
    """
    Removes empty polygons from the string

    :param str fov_polygon: field-of-view polygon based on WKT format
    :return: corrected field-of-view polygon string
    :rtype: str
    ::

        corrected_fov_polygon = correct_fov_polygon(fov_polygon)
    """
    fov_polygon = fov_polygon.replace("(),","")
    fov_polygon = fov_polygon.replace("(()),","")
    fov_polygon = fov_polygon.replace("((())),","")

    return fov_polygon


def plot_ploygons(image_map: np.array, fov_polygon: str, translation_to_global_coordinates: Dict[str, float], scale_factor: float, map_height: int) -> np.array:
    """
    Plots polygons

    :param np.array image_map: image of floor plan
    :param str fov_polygon: field-of-view polygon based on WKT format
    :param Dict[str, float] translation_to_global_coordinates: translation to global coordinates
    :param float scale_factor: scale factor
    :param int map_height: frame height
    :return: plotted image
    :rtype: np.array
    ::

        plotted_image = plot_ploygons(image_map, fov_polygon, translation_to_global_coordinates, scale_factor, map_height)
    """
    shapely_object = wkt.loads(correct_fov_polygon(fov_polygon))

    if fov_polygon.startswith("POLYGON"):
        polygon = shapely_object
        coord_array: List[np.array] = list()
        exterior_coords = list(polygon.exterior.coords)
        exterior_coords = [convert_to_map_pixel(coord, translation_to_global_coordinates, scale_factor, map_height) for coord in exterior_coords]
        coord_array.append(np.array(exterior_coords, dtype=np.int32))
        del exterior_coords

        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            interior_coords = [convert_to_map_pixel(coord, translation_to_global_coordinates, scale_factor, map_height) for coord in interior_coords]
            coord_array.append(np.array(interior_coords, dtype=np.int32))
            del interior_coords
        cv2.fillPoly(image_map, coord_array, (0, 255, 0))
        del coord_array

    if fov_polygon.startswith("MULTIPOLYGON"):
        for polygon in shapely_object.geoms:
            coord_array: List[np.array] = list()
            exterior_coords = list(polygon.exterior.coords)
            exterior_coords = [convert_to_map_pixel(coord, translation_to_global_coordinates, scale_factor, map_height) for coord in exterior_coords]
            coord_array.append(np.array(exterior_coords, dtype=np.int32))
            del exterior_coords

            for interior in polygon.interiors:
                interior_coords = list(interior.coords)
                interior_coords = [convert_to_map_pixel(coord, translation_to_global_coordinates, scale_factor, map_height) for coord in interior_coords]
                coord_array.append(np.array(interior_coords, dtype=np.int32))
                del interior_coords
            cv2.fillPoly(image_map, coord_array, (0, 255, 0))
            del coord_array

    return image_map


def plot_sensor_fov(image_map: np.array, sensor_state_object: SensorStateObject, radius: float = 200, half_span: float = 40) -> np.array:
    """
    Plots sensor FOV

    :param np.array image_map: image of floor plan
    :param SensorStateObject sensor_state_object: sensor state object
    :param float radius: radius of the fan
    :param float half_span: half span of the fan in degree
    :return: plotted image
    :rtype: np.array
    ::

        plotted_image = plot_sensor_fov(image_map, sensor_state_object, radius, half_span)
    """
    image_copy = np.copy(image_map)
    coordinates = sensor_state_object.sensor.coordinates
    coordinates = convert_to_map_pixel([coordinates["x"], coordinates["y"]], sensor_state_object.sensor.translationToGlobalCoordinates,
                                       sensor_state_object.sensor.scaleFactor, image_map.shape[0])
    start_angle = sensor_state_object.direction - half_span
    end_angle = sensor_state_object.direction + half_span
    image_copy = plot_fan_shape(image_copy, coordinates, start_angle, end_angle, radius)

    fov_polygon = sensor_state_object.fieldOfViewPolygon
    if len(fov_polygon) > 0:
        image_copy = plot_ploygons(image_copy, fov_polygon, sensor_state_object.sensor.translationToGlobalCoordinates,
                                   sensor_state_object.sensor.scaleFactor, image_map.shape[0])

    return image_copy


def blend_overlaid_images(image_map: np.array, alpha: float, overlaid_images: List[np.array]) -> np.array:
    """
    Blend overlaid images

    :param np.array image_map: image of floor plan
    :param float alpha: ratio of input image
    :param List[np.array] overlaid_images: overlaid images
    :return: blended image
    :rtype: np.array
    ::

        blended_image = blend_overlaid_images(image_map, alpha, overlaid_images)
    """
    overlaid_image_weight = (1. - alpha) / len(overlaid_images)

    output_image = np.zeros_like(image_map)
    output_image = output_image + (image_map * alpha)

    for overlaid_image in overlaid_images:
        output_image = output_image + (overlaid_image * overlaid_image_weight)

    output_image = output_image.astype(np.uint8)

    return output_image


def plot_sensor_icon_and_fov(image_map: np.array, sensor_state_objects: List[Optional[SensorStateObject]]):
    """
    Plots sensor icon and FOV

    :param np.array image_map: image of floor plan
    :param List[Optional[SensorStateObject]] sensor_state_objects: list of sensor state objects or None
    :return: plotted image
    :rtype: np.array
    ::

        plotted_image = plot_sensor_icon_and_fov(image_map, sensor_state_objects)
    """
    image_copy = np.copy(image_map)
    image_fov = np.zeros(image_copy.shape)
    
    if len(sensor_state_objects) > 0:
        for sensor_state_object in sensor_state_objects:
            if sensor_state_object is not None:
                image_copy = plot_sensor_icon(image_copy, sensor_state_object)
                image_fov = plot_sensor_fov(image_fov, sensor_state_object)            
        alpha = 0.1
        image_fov = np.asarray(image_fov, np.uint8)
        cv2.addWeighted(image_fov, alpha, image_copy, 1 - alpha, 0, image_copy)
        del image_fov

    return pad_image(image_copy)


def plot_overlaid_map(image_map: np.array, sensor_state_objects: List[Optional[SensorStateObject]], 
                      global_people: GlobalObjects, global_amrs: GlobalObjects,
                      rtls_log: List[str], frame_ids: List[int], amr_log: List[str], amr_frame_ids: List[int], 
                      frame_id: int, frame_id_offset: int = 0) -> np.array:
    """
    Plots overlaid information on a map image

    :param np.array image_map: image of floor plan
    :param List[Optional[SensorStateObject]] sensor_state_objects: list of sensor state objects or None
    :param GlobalObjects global_people: global person objects
    :param GlobalObjects global_amrs: global AMR objects
    :param List[str] rtls_log: RTLS log
    :param List[int] frame_ids: frame IDs
    :param List[str] amr_log: AMR log
    :param List[int] amr_frame_ids: AMR frame IDs
    :param int frame_id: frame ID
    :param int frame_id_offset: frame ID offset
    :return: plotted image
    :rtype: np.array
    ::

        plotted_image = plot_overlaid_map(image_map, sensor_state_objects, global_people, global_amrs, rtls_log, frame_ids, amr_log, amr_frame_ids, frame_id, frame_id_offset)
    """
    frame_id += frame_id_offset

    image_copy = np.copy(image_map)
    image_copy = plot_sensor_icon_and_fov(image_copy, sensor_state_objects)

    cv2.putText(image_copy, "Multi-Camera Tracking", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_copy, "Multi-Camera Tracking", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    ### Update and draw people trajectories ###
    if len(rtls_log) > 0:
        target_frame_idx = bisect.bisect(frame_ids, frame_id) - 1
        if target_frame_idx >= 0:

            line = rtls_log[target_frame_idx]
            line = line.rstrip()
            # Remove the "b" symbol (byte) to read the line correctly 
            if line.startswith("b'"):
                line = line[2:-1]
            rtls_log_line = json.loads(line)
    
            object_count = 0
            for i in range(len(rtls_log_line["objectCounts"])):
                if rtls_log_line["objectCounts"][i]["type"] == "Person":
                    object_count = rtls_log_line["objectCounts"][i]["count"]
                    break
            cv2.putText(image_copy, "#people: {}".format(object_count), (940, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image_copy, "#people: {}".format(object_count), (940, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            timestamp = rtls_log_line["timestamp"]

            locations_of_objects = rtls_log_line["locationsOfObjects"]
            target_frame_id = frame_ids[target_frame_idx]
            global_people.update(locations_of_objects, target_frame_id, timestamp)
            map_global_id_to_trajectory_info = global_people.get_trajectory_info()

            for global_id in map_global_id_to_trajectory_info.keys():
                color = map_global_id_to_trajectory_info[global_id]["color"]
                trajectory = map_global_id_to_trajectory_info[global_id]["trajectory"]
                x, y = trajectory[-1]
                radius = 9

                cv2.circle(image_copy, (x, y), radius, (255, 255, 255), thickness=-1)
                cv2.circle(image_copy, (x, y), radius, color, thickness=3)
                cv2.polylines(image_copy, np.int32([np.array(trajectory).reshape(-1, 1, 2)]), False, color, 2)
                cv2.putText(image_copy, str(global_id), (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
                cv2.putText(image_copy, str(global_id), (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
   
    ### Update and draw AMR trajectories ###
    if len(amr_log) > 0:
        target_frame_idx_amr = bisect.bisect(amr_frame_ids, frame_id) - 1
        if target_frame_idx_amr >= 0:

            line = amr_log[target_frame_idx_amr]
            line = line.rstrip()
            # Remove the "b" symbol (byte) to read the line correctly 
            if line.startswith("b'"):
                line = line[2:-1]
            amr_log_line = json.loads(line)

            amr_count = 0
            for i in range(len(amr_log_line["objectCounts"])):
                if amr_log_line["objectCounts"][i]["type"] == "AMR":
                    amr_count = amr_log_line["objectCounts"][i]["count"]
                    break
            cv2.putText(image_copy, "#AMRs: {}".format(amr_count), (1200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image_copy, "#AMRs: {}".format(amr_count), (1200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            timestamp = amr_log_line["timestamp"]

            locations_of_amrs = amr_log_line["locationsOfObjects"]
            target_frame_id_amr = amr_frame_ids[target_frame_idx_amr]
            global_amrs.update(locations_of_amrs, target_frame_id_amr, timestamp)
            map_global_id_to_trajectory_info = global_amrs.get_trajectory_info()

            for global_id in map_global_id_to_trajectory_info.keys():
                trajectory = map_global_id_to_trajectory_info[global_id]["trajectory"]
                x, y = trajectory[-1]

                color = (0,0,0)
                image_copy = plot_amr_icon(image_copy, (x, y))
                cv2.polylines(image_copy, np.int32([np.array(trajectory).reshape(-1, 1, 2)]), False, color, 3)
                cv2.polylines(image_copy, np.int32([np.array(trajectory).reshape(-1, 1, 2)]), False, (255,255,255), 1)
                cv2.putText(image_copy, str(global_id), (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
                cv2.putText(image_copy, str(global_id), (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return pad_image(image_copy)


def plot_amr_icon(image_map: np.array, center: Tuple[int, int]) -> np.array:
    """
    Plots "+" shape icon representing an AMR on image

    :param np.array image_map: image of floor plan
    :param Tuple[int] center: center location of AMR icon
    :return: plotted image
    :rtype: np.array
    ::

        plotted_image = plot_amr_icon(image_map, center)
    """
    cv2.ellipse(image_map, center, (11, 2), 0, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(image_map, center, (11, 2), 90, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(image_map, center, (5, 1), 0, 0, 360, (255, 255, 255), 2)
    cv2.ellipse(image_map, center, (5, 1), 90, 0, 360, (255, 255, 255), 2)

    return image_map


def plot_overlaid_frame(image_frame: np.array, data_dict: Dict[int, Dict[str, Dict[str, Any]]], frame_id: int, sensor_id: str,
                        padding_width: int = 3, padding_color: Tuple[int, int, int] = (255, 255, 255), enable_darkening_image: bool = False) -> np.array:
    """
    Plots overlaid information on a frame image

    :param np.array image_frame: image of frame
    :param Dict[int,Dict[str,Dict[str,Any]]] data_dict: dictionary of protobuf/JSON data
    :param int frame_id: frame ID
    :param str sensor_id: sensor ID
    :param int padding_width: padding width
    :param Tuple[int,int,int] padding_color: padding color
    :param bool enable_darkening_image: flag indicating whether to darken image
    :return: plotted image
    :rtype: np.array
    ::

        plotted_image = plot_overlaid_frame(image_frame, data_dict, frame_id, sensor_id, padding_width, padding_color, enable_darkening_image)
    """
    bbox_color = (0, 255, 0)
    thickness = 3
    image_copy = np.copy(image_frame)

    if frame_id in data_dict.keys():
        map_sensor_id_to_objects_dict = data_dict[frame_id]
        if sensor_id in map_sensor_id_to_objects_dict:
            map_object_id_to_info = map_sensor_id_to_objects_dict[sensor_id]
            for object_id in map_object_id_to_info:
                bbox, foot_location = map_object_id_to_info[object_id]
                left_x, top_y, right_x, bottom_y = bbox
                left_x = int(left_x)
                top_y = int(top_y)
                right_x = int(right_x)
                bottom_y = int(bottom_y)
                cv2.rectangle(image_copy, (left_x, top_y), (right_x, bottom_y), bbox_color, thickness)
                cv2.putText(image_copy, "{}".format(object_id), (left_x, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, bbox_color, 3, cv2.LINE_AA)
                image_copy = cv2.circle(image_copy, (int(foot_location[0]), int(foot_location[1])), 5, (255, 0, 0), thickness=-1)
                image_copy = cv2.circle(image_copy, (int(foot_location[0]), int(foot_location[1])), 3, (255, 255, 255), thickness=-1)

    image_copy = cv2.putText(image_copy, "{}".format(sensor_id.split("_")[-1]), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
    image_copy = cv2.putText(image_copy, "{}".format(sensor_id.split("_")[-1]), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    if enable_darkening_image:
        image_copy = darken_image(image_copy, alpha=0.5)

    return pad_image(image_copy, boundary_width=padding_width, color=padding_color)


def plot_combined_image(config: VizConfig, image_map: np.array, map_video_name_to_capture: Dict[str, cv2.VideoCapture], global_people: GlobalObjects, global_amrs: GlobalObjects,
                        data_dict: Dict[int, Dict[str, Dict[str, Any]]], rtls_log: List[str], frame_ids: List[int], amr_log: List[str], amr_frame_ids: List[int],
                        frame_id: int, read_frame_only: bool = False) -> np.array:
    """
    Plots combined image

    :param VizConfig config: visualization config
    :param np.array image_map: image of floor plan
    :param Dict[str,cv2.VideoCapture] map_video_name_to_capture: map from video names to video captures
    :param GlobalObjects global_people: global person objects
    :param GlobalObjects global_amrs: global AMR objects
    :param Dict[int,Dict[str,Dict[str,Any]]] data_dict: dictionary of protobuf/JSON data
    :param List[str] rtls_log: RTLS log
    :param List[int] frame_ids: frame IDs
    :param List[str] amr_log: AMR log
    :param List[int] amr_frame_ids: AMR frame IDs
    :param int frame_id: frame ID
    :param bool read_frame_only: flag indicating whether to ready frame only
    :return: plotted image
    :rtype: np.array
    ::

        plotted_image = plot_overlaid_frame(config, image_map, map_video_name_to_capture, global_people, global_amrs,
                                            data_dict, rtls_log, frame_ids, amr_log, amr_frame_ids, frame_id, read_frame_only)
    """
    if read_frame_only:
        for video_name in map_video_name_to_capture:
            video_capture = map_video_name_to_capture[video_name]
            _, image_frame = video_capture.read()

    else:
        activated_sensor_id = None
        start_time_sec = config.rtls_config.output.sensorViewStartTimeSec
        duration_sec = config.rtls_config.output.sensorViewDurationSec
        gap_sec = config.rtls_config.output.sensorViewGapSec

        sensor_ids = list(sorted(config.sensor_state_objects.keys()))
        num_sensors = len(sensor_ids)
        sensor_activation_frame_range: Dict[str, Tuple[int, int]] = dict()
        for i in range(num_sensors):
            sensor_activation_frame_range[sensor_ids[i]] = ((start_time_sec + (i * duration_sec) + (i * gap_sec)) * config.fps,
                                                            (start_time_sec + ((i + 1) * duration_sec) + (i * gap_sec)) * config.fps)

        image_frames: List[np.array] = []
        if config.rtls_config.output.displaySensorViews:
            for sensor_id in sensor_activation_frame_range.keys():
                start_frame_id, end_frame_id = sensor_activation_frame_range[sensor_id]
                if start_frame_id < frame_id < end_frame_id:
                    activated_sensor_id = sensor_id
                    if sensor_id not in config.activated_sensor_ids:
                        config.activated_sensor_ids.append(sensor_id)
            for video_name in map_video_name_to_capture:
                video_capture = map_video_name_to_capture[video_name]
                _, image_frame = video_capture.read()
                sensor_id = video_name.split(".mp4")[0]
                if ((config.rtls_config.output.sensorViewDisplayMode.value == "rotational") and (sensor_id == activated_sensor_id)) or \
                    ((config.rtls_config.output.sensorViewDisplayMode.value == "cumulative") and (sensor_id in config.activated_sensor_ids)):
                    image_frame = plot_overlaid_frame(image_frame, data_dict, frame_id, sensor_id, padding_width=12, padding_color=(0, 255, 0))
                else: 
                    if (frame_id < start_time_sec * config.fps) or (frame_id >= (start_time_sec + (num_sensors * duration_sec) + ((num_sensors - 1) * gap_sec)) * config.fps):
                        image_frame = plot_overlaid_frame(image_frame, data_dict, frame_id, sensor_id)
                    else:
                        image_frame = plot_overlaid_frame(image_frame, data_dict, frame_id, sensor_id, enable_darkening_image=True)
                image_frames.append(image_frame)

        sensor_state_objects: List[str] = list()
        if config.rtls_config.output.sensorFovDisplayMode.value == 'rotational':
            sensor_state_objects = [config.sensor_state_objects.get(activated_sensor_id, None)]
        elif config.rtls_config.output.sensorFovDisplayMode.value == 'cumulative':
            sensor_state_objects = [config.sensor_state_objects.get(activated_sensor_id, None) for activated_sensor_id in config.activated_sensor_ids]

        image_map = plot_overlaid_map(image_map, sensor_state_objects, global_people, global_amrs, rtls_log, frame_ids, amr_log, amr_frame_ids, frame_id)
        image_map = cv2.resize(image_map, (config.output_map_width, config.output_map_height), interpolation=cv2.INTER_LINEAR)
        image_output = image_map

        if config.rtls_config.output.displaySensorViews:
            # Assume all frame images share the same size
            black_patch = np.zeros([image_frames[0].shape[0], image_frames[0].shape[1], 3], dtype=np.uint8)

            if len(image_frames) < num_sensors:
                num_black_patches = num_sensors - len(image_frames)
                image_frames += [black_patch] * num_black_patches

            if num_sensors == 8:
                if config.sensor_views_layout == "radial":
                    image_left = np.vstack([image_frames[7], image_frames[6], image_frames[5], image_frames[1]])
                    image_right = np.vstack([image_frames[2], image_frames[3], image_frames[4], image_frames[0]])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_right = cv2.resize(image_right, (int(image_right.shape[1] * float(image_map.shape[0]) / float(image_right.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map, image_right])
                elif config.sensor_views_layout == "split":
                    image_left = np.hstack([np.vstack(image_frames[:4]), np.vstack(image_frames[4:8])])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map])

            if num_sensors == 12:
                if config.sensor_views_layout == "radial":
                    image_left = np.hstack([np.vstack(image_frames[:3]), np.vstack(image_frames[3:6])])        
                    image_right = np.hstack([np.vstack(image_frames[6:9]), np.vstack(image_frames[9:12])])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_right = cv2.resize(image_right, (int(image_right.shape[1] * float(image_map.shape[0]) / float(image_right.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map, image_right])
                elif config.sensor_views_layout == "split":
                    image_left = np.hstack([np.vstack(image_frames[:4]), np.vstack(image_frames[4:8]), np.vstack(image_frames[8:12])])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map])

            if num_sensors == 16:
                if config.sensor_views_layout == "radial":
                    image_left = np.hstack([np.vstack(image_frames[:4]), np.vstack(image_frames[4:8])])        
                    image_right = np.hstack([np.vstack(image_frames[8:12]), np.vstack(image_frames[12:16])])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_right = cv2.resize(image_right, (int(image_right.shape[1] * float(image_map.shape[0]) / float(image_right.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map, image_right])
                elif config.sensor_views_layout == "split":
                    image_left = np.hstack([np.vstack(image_frames[:4]), np.vstack(image_frames[4:8]), np.vstack(image_frames[8:12]), np.vstack(image_frames[12:16])])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map])

            if num_sensors == 30:
                if config.sensor_views_layout == "radial":
                    image_left = np.hstack([np.vstack(image_frames[:5]), np.vstack(image_frames[5:10]), np.vstack(image_frames[10:15])])        
                    image_right = np.hstack([np.vstack(image_frames[15:20]), np.vstack(image_frames[20:25]), np.vstack(image_frames[25:30])])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_right = cv2.resize(image_right, (int(image_right.shape[1] * float(image_map.shape[0]) / float(image_right.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map, image_right])
                elif config.sensor_views_layout == "split":
                    image_left = np.hstack([np.vstack(image_frames[:6]), np.vstack(image_frames[6:12]), np.vstack(image_frames[12:18]), np.vstack(image_frames[18:24]), np.vstack(image_frames[24:30])])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map])

            if num_sensors == 40:
                if config.sensor_views_layout == "radial":
                    image_left_a = np.vstack([black_patch, black_patch] + image_frames[:3] + [black_patch, black_patch])
                    image_left_b = np.vstack([black_patch] + image_frames[3:8] + [black_patch])
                    image_left_c = np.vstack(image_frames[8:15])
                    image_left = np.hstack([image_left_a, image_left_b, image_left_c])
                    image_mid_top = np.hstack(image_frames[15:20])
                    image_mid_bottom = np.hstack(image_frames[20:25])
                    image_mid_top = cv2.resize(image_mid_top, (image_map.shape[1], int(image_mid_top.shape[0] * float(image_map.shape[1]) / float(image_mid_top.shape[1]))), interpolation=cv2.INTER_LINEAR)
                    image_mid_bottom = cv2.resize(image_mid_bottom, (image_map.shape[1], int(image_mid_bottom.shape[0] * float(image_map.shape[1]) / float(image_mid_bottom.shape[1]))), interpolation=cv2.INTER_LINEAR)
                    image_mid = np.vstack([image_mid_top, image_map, image_mid_bottom])
                    image_right_a = np.vstack(image_frames[25:32])
                    image_right_b = np.vstack([black_patch] + image_frames[32:37] + [black_patch])
                    image_right_c = np.vstack([black_patch, black_patch] + image_frames[37:40] + [black_patch, black_patch])
                    image_right = np.hstack([image_right_a, image_right_b, image_right_c])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_mid.shape[0]) / float(image_left.shape[0])), image_mid.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_right = cv2.resize(image_right, (int(image_right.shape[1] * float(image_mid.shape[0]) / float(image_right.shape[0])), image_mid.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_mid, image_right])
                elif config.sensor_views_layout == "split":
                    image_left = np.hstack([np.vstack(image_frames[:8]), np.vstack(image_frames[8:16]), np.vstack(image_frames[16:24]), np.vstack(image_frames[24:32]), np.vstack(image_frames[32:40])])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map])

            if num_sensors == 96:
                if config.sensor_views_layout == "radial":
                    image_left_a = np.vstack([black_patch] + image_frames[:10] + [black_patch])
                    image_left_b = np.vstack([black_patch] + image_frames[10:20] + [black_patch])
                    image_left_c = np.vstack(image_frames[20:32])
                    image_left = np.hstack([image_left_a, image_left_b, image_left_c])
                    image_mid_top_a = np.hstack(image_frames[32:40])
                    image_mid_top_b = np.hstack(image_frames[40:48])
                    image_mid_top = np.vstack([image_mid_top_a, image_mid_top_b])
                    image_mid_bottom_a = np.hstack(image_frames[48:56])
                    image_mid_bottom_b = np.hstack(image_frames[56:64])
                    image_mid_bottom = np.vstack([image_mid_bottom_a, image_mid_bottom_b])
                    image_mid_top = cv2.resize(image_mid_top, (image_map.shape[1], int(image_mid_top.shape[0] * float(image_map.shape[1]) / float(image_mid_top.shape[1]))), interpolation=cv2.INTER_LINEAR)
                    image_mid_bottom = cv2.resize(image_mid_bottom, (image_map.shape[1], int(image_mid_bottom.shape[0] * float(image_map.shape[1]) / float(image_mid_bottom.shape[1]))), interpolation=cv2.INTER_LINEAR)
                    image_mid = np.vstack([image_mid_top, image_map, image_mid_bottom])
                    image_right_a = np.vstack(image_frames[64:76])
                    image_right_b = np.vstack([black_patch] + image_frames[76:86] + [black_patch])
                    image_right_c = np.vstack([black_patch] + image_frames[86:96] + [black_patch])
                    image_right = np.hstack([image_right_a, image_right_b, image_right_c])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_mid.shape[0]) / float(image_left.shape[0])), image_mid.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_right = cv2.resize(image_right, (int(image_right.shape[1] * float(image_mid.shape[0]) / float(image_right.shape[0])), image_mid.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_mid, image_right])
                elif config.sensor_views_layout == "split":
                    image_left = np.hstack([np.vstack(image_frames[i:i+12]) for i in range(0, 96, 12)])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map])

            if num_sensors == 100:
                if config.sensor_views_layout == "radial":
                    image_left_a = np.vstack([black_patch] + image_frames[:10] + [black_patch])
                    image_left_b = np.vstack(image_frames[10:22])
                    image_left_c = np.vstack(image_frames[22:34])
                    image_left = np.hstack([image_left_a, image_left_b, image_left_c])
                    image_mid_top_a = np.hstack(image_frames[34:42])
                    image_mid_top_b = np.hstack(image_frames[42:50])
                    image_mid_top = np.vstack([image_mid_top_a, image_mid_top_b])
                    image_mid_bottom_a = np.hstack(image_frames[50:58])
                    image_mid_bottom_b = np.hstack(image_frames[58:66])
                    image_mid_bottom = np.vstack([image_mid_bottom_a, image_mid_bottom_b])
                    image_mid_top = cv2.resize(image_mid_top, (image_map.shape[1], int(image_mid_top.shape[0] * float(image_map.shape[1]) / float(image_mid_top.shape[1]))), interpolation=cv2.INTER_LINEAR)
                    image_mid_bottom = cv2.resize(image_mid_bottom, (image_map.shape[1], int(image_mid_bottom.shape[0] * float(image_map.shape[1]) / float(image_mid_bottom.shape[1]))), interpolation=cv2.INTER_LINEAR)
                    image_mid = np.vstack([image_mid_top, image_map, image_mid_bottom])
                    image_right_a = np.vstack(image_frames[66:78])
                    image_right_b = np.vstack(image_frames[78:90])
                    image_right_c = np.vstack([black_patch] + image_frames[90:100] + [black_patch])
                    image_right = np.hstack([image_right_a, image_right_b, image_right_c])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_mid.shape[0]) / float(image_left.shape[0])), image_mid.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_right = cv2.resize(image_right, (int(image_right.shape[1] * float(image_mid.shape[0]) / float(image_right.shape[0])), image_mid.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_mid, image_right])
                elif config.sensor_views_layout == "split":
                    image_left = np.hstack([np.vstack(image_frames[i:i+10]) for i in range(0, 100, 10)])
                    image_left = cv2.resize(image_left, (int(image_left.shape[1] * float(image_map.shape[0]) / float(image_left.shape[0])), image_map.shape[0]), interpolation=cv2.INTER_LINEAR)
                    image_output = np.hstack([image_left, image_map])

        return image_output
