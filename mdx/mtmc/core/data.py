# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import logging
import ast
import numpy as np
import matplotlib.path as mplPath
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from mdx.mtmc.config import AppConfig
from mdx.mtmc.schema import Bbox, Object, Frame, Behavior, SensorStateObject
from mdx.mtmc.stream.state.people_height import StateManager as PeopleHeightEstimator
from mdx.mtmc.utils.io_utils import validate_file_path, load_json_from_str

FOOT_PIXEL_MARGIN = 0.1


def calculate_bbox_area(bbox: Bbox) -> float:
    """
    Calculates bbox area

    :param Bbox bbox: bounding box
    :return: area of the bbox
    :rtype: float
    """
    bbox_width = bbox.rightX - bbox.leftX + 1.
    bbox_height = bbox.bottomY - bbox.topY + 1.
    return bbox_width * bbox_height


def calculate_bbox_aspect_ratio(bbox: Bbox) -> float:
    """
    Calculates bbox aspect ratio

    :param Bbox bbox: bounding box
    :return: area of the bbox
    :rtype: float
    """
    bbox_width = bbox.rightX - bbox.leftX + 1.
    bbox_height = bbox.bottomY - bbox.topY + 1.
    return bbox_width / bbox_height


def perspective_transform(pixel: List[float], homography: List[List[float]]) -> Optional[List[float]]:
    """
    Transforms a 2D pixel to a location on the ground plane

    :param List[float] pixel: 2D pixel location
    :param List[List[float]] homography: 3x3 homography matrix
    :return: location on the ground plane
    :rtype: Optional[List[float]]
    """
    x = pixel[0]
    y = pixel[1]
    w = homography[2][0] * x + homography[2][1] * y + homography[2][2]
    if w == 0:
        return None
    transformed_x = (homography[0][0] * x + homography[0][1] * y + homography[0][2]) / w
    transformed_y = (homography[1][0] * x + homography[1][1] * y + homography[1][2]) / w
    return [transformed_x, transformed_y]


def normalize_vector(vector: List[float]) -> Optional[np.array]:
    """
    Normalizes a vector

    :param List[float] vector: vector
    :return: normalized vector or None
    :rtype: Optional[List[float]]
    """
    if len(vector) > 0:
        vector_norm = np.linalg.norm(vector)
        if vector_norm > 0.:
            return vector / vector_norm
        else:
            logging.warning(f"WARNING: The norm of the input vector for normalization is zero. None is returned.")
    return None


class Loader:
    """
    Module to load data

    :param AppConfig config: configuration for the app
    ::

        loader = Loader(config)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self.selected_sensor_ids = set(config.io.selectedSensorIds)

    def load_json_data_to_frames(self, json_data_path: str) -> List[Frame]:
        """
        Loads JSON data from the perception pipeline

        :param str json_data_path: path to the JSON data file
        :return: list of frames
        :rtype: List[Frame]
        ::

            frames = loader.load_json_data_to_frames(json_data_path)
        """
        json_data_path = validate_file_path(json_data_path)
        if not os.path.isfile(json_data_path):
            logging.error(f"ERROR: The JSON data path {json_data_path} does NOT exist.")
            exit(1)
        logging.info(f"Loading JSON data from {json_data_path}...")

        frames: List[Frame] = list()
        object_count = 0

        with open(json_data_path, "r") as f:
            for line in f.readlines():
                line = line.rstrip('\n').replace("'", '"')
                raw_data = load_json_from_str(line)

                version = raw_data["version"]
                frame_id = raw_data["id"]
                timestamp = raw_data["@timestamp"]
                sensor_id = raw_data["sensorId"]

                if (len(self.selected_sensor_ids) > 0) and (sensor_id not in self.selected_sensor_ids):
                    continue

                objects: List[Object] = list()

                for raw_object in raw_data["objects"]:
                    object_tokens = raw_object.split("|")

                    # Only consider person objects
                    object_type = object_tokens[5]
                    if object_type not in {"Person"}:
                        continue

                    track_id = object_tokens[0]
                    bbox = Bbox(leftX=float(object_tokens[1]), topY=float(object_tokens[2]),
                                rightX=float(object_tokens[3]), bottomY=float(object_tokens[4]))

                    object_info: Dict[str, str] = dict()
                    if object_type == "Person":
                        if (object_tokens[6] == "#"):
                            object_info["gender"] = object_tokens[7]
                            object_info["age"] = object_tokens[8]
                            object_info["hair"] = object_tokens[9]
                            object_info["cap"] = object_tokens[10]
                            object_info["apparel"] = object_tokens[11]
                        else:
                            logging.error(f"ERROR: Missing secondary attributes for the object {raw_object}.")
                            exit(1)

                    confidence = 1.
                    if len(object_tokens) > 12:
                        # Handle empty confidence
                        try:
                            confidence = float(object_tokens[12])
                        except ValueError:
                            pass

                    embedding = None
                    try:
                        embedding_idx = object_tokens.index("embedding")
                        embedding = object_tokens[embedding_idx+1]
                        embedding = [float(x) for x in embedding.split(",")]
                        # Normalize the embedding vector
                        embedding = normalize_vector(embedding)
                        if embedding is not None:
                            embedding = embedding.tolist()
                    except ValueError:
                        pass

                    try:
                        sv3dt_idx = object_tokens.index("SV3DT")
                        object_info["visibility"] = object_tokens[sv3dt_idx+1]
                        object_info["footLocation"] = object_tokens[sv3dt_idx+2]
                        object_info["footLocation3D"] = object_tokens[sv3dt_idx+3]
                        object_info["convexHull"] = object_tokens[sv3dt_idx+4]
                    except ValueError:
                        pass

                    object_instance = Object(id=track_id, bbox=bbox, type=object_type, confidence=confidence,
                                             info=object_info, embedding=embedding)
                    del embedding
                    objects.append(object_instance)
                    object_count += 1

                frame = Frame(version=version, id=frame_id, timestamp=timestamp, sensorId=sensor_id, objects=objects)
                frames.append(frame)

        logging.info(f"No. frames: {len(frames)}")
        logging.info(f"No. objects: {object_count}")

        return frames

    def load_protobuf_string_to_frame(self, protobuf_string: str) -> Optional[Frame]:
        """
        Loads a protobuf string to a frame object

        :param str protobuf_string: protobuf string
        :return: frame object or None
        :rtype: Optional[Frame]
        ::

            frame = loader.load_protobuf_string_to_frame(protobuf_string)
        """
        protobuf_string = protobuf_string.rstrip('\n').replace("'", '"')
        json_data = load_json_from_str(protobuf_string)

        if "locationsOfObjects" in json_data.keys():
            return None

        sensor_id = json_data["sensorId"]
        if (len(self.selected_sensor_ids) > 0) and (sensor_id not in self.selected_sensor_ids):
            return None

        for object in json_data["objects"]:
            if "embedding" in object and "vector" in object["embedding"]:
                object["embedding"] = object["embedding"]["vector"]
            else:
                object["embedding"] = None

        return Frame(**json_data)

    def load_protobuf_message_to_frame(self, protobuf_frame: Any) -> Optional[Frame]:
        """
        Loads a protobuf frame object to a frame object

        :param Any frame: protobuf frame object
        :return: frame object or None
        :rtype: Optional[Frame]
        ::

            frame = loader.load_protobuf_message_to_frame(protobuf_frame)
        """
        sensor_id = protobuf_frame.sensorId
        if (len(self.selected_sensor_ids) > 0) and (sensor_id not in self.selected_sensor_ids):
            return None

        version = protobuf_frame.version
        frame_id = protobuf_frame.id

        timestamp_ms = int((protobuf_frame.timestamp.seconds + (protobuf_frame.timestamp.nanos * (10 ** -9))) * 1000)
        timestamp_str = f"{datetime.utcfromtimestamp(timestamp_ms / 1000).isoformat(timespec='milliseconds')}Z"
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

        objects: List[Object] = list()

        for object_instance in protobuf_frame.objects:
            track_id = object_instance.id
            object_type = object_instance.type

            confidence = object_instance.confidence
            # Handle empty confidence
            if confidence == 0.:
                confidence = 1.

            object_info = object_instance.info

            bbox = Bbox(
                leftX=object_instance.bbox.leftX,
                topY=object_instance.bbox.topY,
                rightX=object_instance.bbox.rightX,
                bottomY=object_instance.bbox.bottomY
            )

            embedding = None
            if hasattr(object_instance, "embedding") and hasattr(object_instance.embedding, "vector"):
                embedding = normalize_vector(object_instance.embedding.vector)
                if embedding is not None:
                    embedding = embedding.tolist()

            objects.append(Object(id=track_id, bbox=bbox, type=object_type, confidence=confidence,
                                  info=object_info, embedding=embedding))
            del object_info
            del bbox
            del embedding

        frame = Frame(version=version, id=frame_id, timestamp=timestamp, sensorId=sensor_id, objects=objects)
        del objects

        return frame

    def load_protobuf_data_to_frames(self, protobuf_data_path: str) -> List[Frame]:
        """
        Loads protobuf data from the perception pipeline

        :param str protobuf_data_path: path to the protobuf data file
        :return: list of frames
        :rtype: List[Frame]
        ::

            frames = loader.load_protobuf_data_to_frames(protobuf_data_path)
        """
        protobuf_data_path = validate_file_path(protobuf_data_path)
        if not os.path.isfile(protobuf_data_path):
            logging.error(f"ERROR: The protobuf data path {protobuf_data_path} does NOT exist.")
            exit(1)
        logging.info(f"Loading protobuf data from {protobuf_data_path}...")

        frames: List[Frame] = list()
        object_count = 0

        with open(protobuf_data_path, "r") as f:
            for line in f.readlines():
                frame = self.load_protobuf_string_to_frame(line)
                # Uncomment if the input protobuf file is in byte instead of in text
                # frame = self.load_protobuf_message_to_frame(ast.literal_eval(line.strip()))
                if frame is None:
                    continue
                frames.append(frame)
                object_count += len(frame.objects)

        logging.info(f"No. frames: {len(frames)}")
        logging.info(f"No. objects: {object_count}")

        return frames


class Preprocessor:
    """
    Module to preprocess frames into behaviors

    :param AppConfig config: configuration for the app
    ::

        preprocessor = Preprocessor(config)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self.sensor_state_objects: Dict[str, SensorStateObject] = dict()
        self.people_height_estimator: Optional[PeopleHeightEstimator] = None

    def _filter_object_by_rois(self, location: List[float], roi_paths: List[mplPath.Path]) -> bool:
        """
        Filters object by regions of interest

        :param List[float] location: location
        :param List[mplPath.Path] roi_paths: paths of regions of interest
        :return: flag indicating whether the location is within the regions of interest
        :rtype: bool
        """
        if len(roi_paths) == 0:
            return False

        for roi_path in roi_paths:
            if roi_path.contains_point(location):
                return False

        return True

    def _filter_by_confidence_and_bbox(self, confidence: float, bbox: Bbox, sensor_id: str, confidence_thresh: float, bbox_aspect_ratio_thresh: float, bbox_area_thresh: float) -> bool:
        """
        Filters instance by confidence and bounding box

        :param float confidence: confidence
        :param Bbox bbox: bounding box
        :param str sensor_id: sensor ID
        :param float confidence_thresh: confidence threshold
        :param float bbox_aspect_ratio_thresh: bounding box aspect ratio threshold
        :param float bbox_area_thresh: bounding box area threshold
        :return: flag indicating whether to filter the instance
        :rtype: bool
        """
        if confidence >= confidence_thresh:
            if calculate_bbox_aspect_ratio(bbox) < bbox_aspect_ratio_thresh:
                frame_width = self.sensor_state_objects[sensor_id].frameWidth
                frame_height = self.sensor_state_objects[sensor_id].frameHeight
                if (frame_width is not None) and (frame_height is not None):
                    if calculate_bbox_area(bbox) > (frame_width * frame_height * bbox_area_thresh):
                        return False
        return True

    def _filter_by_bbox_touching_frame_bottom(self, bbox: Bbox, sensor_id: str, bbox_bottom_gap_thresh: float) -> bool:
        """
        Filters instance by bounding box touching the frame bottom

        :param Bbox bbox: bounding box
        :param str sensor_id: sensor ID
        :param float bbox_bottom_gap_thresh: bounding box bottom gap threshold
        :return: flag indicating whether to filter the instance
        :rtype: bool
        """
        frame_height = self.sensor_state_objects[sensor_id].frameHeight

        if frame_height is None:
            return False

        if bbox.bottomY > (1. - bbox_bottom_gap_thresh) * frame_height:
            return True

        return False

    def _split_behavior(self, behavior: Behavior) -> List[Behavior]:
        """
        Splits a behavior if the neighboring timestamps have a large gap

        :param Behavior: behavior
        :return: behaviors after splitting
        :rtype: List[Behavior]
        """
        # Return if the behavior length is 1 or less
        num_timestamps = len(behavior.timestamps)
        if len(behavior.timestamps) <= 1:
            return [behavior]

        # Sort all the lists based on timestamps
        sorted_indices = np.argsort(np.array(behavior.timestamps))
        behavior.timestamps = [behavior.timestamps[i] for i in sorted_indices]
        behavior.frameIds = [behavior.frameIds[i] for i in sorted_indices]
        behavior.bboxes = [behavior.bboxes[i] for i in sorted_indices]
        behavior.confidences = [behavior.confidences[i] for i in sorted_indices]

        if len(behavior.locations) > 1:
            map_original_to_shortened_indices = {original_idx: shortened_idx for shortened_idx, original_idx in enumerate([i for i, mask in enumerate(behavior.locationMask) if mask])}
            sorted_locations: List[List[float]] = list()
            for original_idx in sorted_indices:
                if original_idx in map_original_to_shortened_indices:
                    shortened_idx = map_original_to_shortened_indices[original_idx]
                    sorted_locations.append(behavior.locations[shortened_idx])
            del map_original_to_shortened_indices
            behavior.locations = sorted_locations
            del sorted_locations
        behavior.locationMask = [behavior.locationMask[i] for i in sorted_indices]

        if len(behavior.embeddings) > 1:
            map_original_to_shortened_indices = {original_idx: shortened_idx for shortened_idx, original_idx in enumerate([i for i, mask in enumerate(behavior.embeddingMask) if mask])}
            sorted_embeddings: List[List[float]] = list()
            for original_idx in sorted_indices:
                if original_idx in map_original_to_shortened_indices:
                    shortened_idx = map_original_to_shortened_indices[original_idx]
                    sorted_embeddings.append(behavior.embeddings[shortened_idx])
            del map_original_to_shortened_indices
            behavior.embeddings = sorted_embeddings
            del sorted_embeddings
        behavior.embeddingMask = [behavior.embeddingMask[i] for i in sorted_indices]

        behavior.roiMask = [behavior.roiMask[i] for i in sorted_indices]

        del sorted_indices

        # Split based on the gap of timestamps
        split_indices: List[int] = list()
        split_indices.append(0)
        for i in range(1, num_timestamps):
            if behavior.timestamps[i] > behavior.timestamps[i-1] + timedelta(seconds=self.config.preprocessing.behaviorSplitThreshSec):
                split_indices.append(i)

        # Return if there is no split
        if len(split_indices) == 1:
            return [behavior]

        # Create segmented behaviors
        behaviors: List[Behavior] = list()
        for i in range(len(split_indices)):
            idx_start = split_indices[i]
            idx_end = num_timestamps
            if i < len(split_indices) - 1:
                idx_end = split_indices[i+1]

            behavior_segmented = behavior.copy()
            behavior_segmented.timestamp = behavior.timestamps[idx_start]
            behavior_segmented.end = behavior.timestamps[idx_end-1]
            behavior_segmented.startFrame = behavior.frameIds[idx_start]
            behavior_segmented.endFrame = behavior.frameIds[idx_end-1]
            behavior_segmented.timestamps = behavior_segmented.timestamps[idx_start:idx_end]
            behavior_segmented.frameIds = behavior_segmented.frameIds[idx_start:idx_end]
            behavior_segmented.bboxes = behavior_segmented.bboxes[idx_start:idx_end]
            behavior_segmented.confidences = behavior_segmented.confidences[idx_start:idx_end]

            if len(behavior_segmented.locations) > 1:
                idx_start_of_locations = 0
                for j in range(idx_start):
                    if behavior_segmented.locationMask[j]:
                        idx_start_of_locations += 1
                idx_end_of_locations = idx_start_of_locations
                for j in range(idx_start, idx_end):
                    if behavior_segmented.locationMask[j]:
                        idx_end_of_locations += 1
                behavior_segmented.locations = behavior_segmented.locations[idx_start_of_locations:idx_end_of_locations]
            behavior_segmented.locationMask = behavior_segmented.locationMask[idx_start:idx_end]

            if len(behavior_segmented.embeddings) > 1:
                idx_start_of_embeddings = 0
                for j in range(idx_start):
                    if behavior_segmented.embeddingMask[j]:
                        idx_start_of_embeddings += 1
                idx_end_of_embeddings = idx_start_of_embeddings
                for j in range(idx_start, idx_end):
                    if behavior_segmented.embeddingMask[j]:
                        idx_end_of_embeddings += 1
                behavior_segmented.embeddings = behavior_segmented.embeddings[idx_start_of_embeddings:idx_end_of_embeddings]
            behavior_segmented.embeddingMask = behavior_segmented.embeddingMask[idx_start:idx_end]

            behavior_segmented.roiMask = behavior_segmented.roiMask[idx_start:idx_end]

            logging.info(f"Split behavior {behavior.id} from {behavior_segmented.timestamp} to {behavior_segmented.end}")

            behaviors.append(behavior_segmented)

        return behaviors

    def _apply_roi_mask(self, behavior: Behavior) -> Optional[Behavior]:
        """
        Applies the RoI mask on the behavior

        :param Behavior: behavior
        :return: behavior after RoI mask is applied
        :rtype: Optional[Behavior]
        """
        if all(behavior.roiMask):
            behavior.roiMask = None
            return behavior

        if not any(behavior.roiMask):
            return None

        behavior.timestamps = [timestamp for timestamp, is_in_roi in zip(behavior.timestamps, behavior.roiMask) if is_in_roi]
        behavior.timestamp = behavior.timestamps[0]
        behavior.end = behavior.timestamps[-1]
        behavior.frameIds = [frame_id for frame_id, is_in_roi in zip(behavior.frameIds, behavior.roiMask) if is_in_roi]
        behavior.startFrame = behavior.frameIds[0]
        behavior.endFrame = behavior.frameIds[-1]
        behavior.bboxes = [bbox for bbox, is_in_roi in zip(behavior.bboxes, behavior.roiMask) if is_in_roi]
        behavior.confidences = [confidence for confidence, is_in_roi in zip(behavior.confidences, behavior.roiMask) if is_in_roi]

        locations: List[List[float]] = list()
        idx_location = 0
        for i in range(len(behavior.roiMask)):
            if behavior.locationMask[i]:
                if behavior.roiMask[i]:
                    locations.append(behavior.locations[idx_location])
                idx_location += 1
        behavior.locations = locations
        behavior.locationMask = [is_valid_location for is_valid_location, is_in_roi in zip(behavior.locationMask, behavior.roiMask) if is_in_roi]

        embeddings: List[List[float]] = list()
        embedding_idx = 0
        for i in range(len(behavior.roiMask)):
            if behavior.embeddingMask[i]:
                if behavior.roiMask[i]:
                    embeddings.append(behavior.embeddings[embedding_idx])
                embedding_idx += 1
        behavior.embeddings = embeddings
        behavior.embeddingMask = [is_valid_embedding for is_valid_embedding, is_in_roi in zip(behavior.embeddingMask, behavior.roiMask) if is_in_roi]

        behavior.roiMask = None
        return behavior

    def calculate_bbox_area(self, bbox: Bbox) -> float:
        """
        Calculates bbox area

        :param Bbox bbox: bounding box
        :return: area of the bbox
        :rtype: float
        """
        bbox_width = bbox.rightX - bbox.leftX + 1.
        bbox_height = bbox.bottomY - bbox.topY + 1.
        return bbox_width * bbox_height

    def calculate_bbox_aspect_ratio(self, bbox: Bbox) -> float:
        """
        Calculates bbox aspect ratio

        :param Bbox bbox: bounding box
        :return: area of the bbox
        :rtype: float
        """
        bbox_width = bbox.rightX - bbox.leftX + 1.
        bbox_height = bbox.bottomY - bbox.topY + 1.
        return bbox_width / bbox_height

    def set_sensor_state_objects(self, sensor_state_objects: Dict[str, SensorStateObject]) -> None:
        """
        Sets sensor state objects

        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :return: None
        ::

            preprocessor.set_sensor_state_objects(sensor_state_objects)
        """
        self.sensor_state_objects = sensor_state_objects
        if self.people_height_estimator is not None:
            self.people_height_estimator.set_sensor_state_objects(self.sensor_state_objects)

    def set_people_height_estimator(self, people_height_estimator: PeopleHeightEstimator) -> None:
        """
        Sets people height estimator

        :param PeopleHeightEstimator people_height_estimator: estimator of people height
        :return: None
        ::

            preprocessor.set_people_height_estimator(people_height_estimator)
        """
        self.people_height_estimator = people_height_estimator
        self.people_height_estimator.set_sensor_state_objects(self.sensor_state_objects)

    def update_people_height(self, behaviors: List[Behavior]) -> None:
        """
        Updates people's height

        :param List[Behavior] behaviors: list of behaviors
        :return: None
        ::

            preprocessor.update_people_height(behaviors)
        """
        self.people_height_estimator.update_people_height(behaviors)

    def _get_roi_paths(self, sensor_id: str) -> List[mplPath.Path]:
        """
        Gets paths of regions of interest

        :param str sensor_id: sensor ID
        :return: paths of regions of interest
        :rtype: List[mplPath.Path]
        """
        roi_paths: List[mplPath.Path] = list()
        if (sensor_id in self.sensor_state_objects.keys()) and \
            (self.sensor_state_objects[sensor_id].rois is not None):
            for roi in self.sensor_state_objects[sensor_id].rois:
                roi_paths.append(mplPath.Path(roi))
        return roi_paths
    
    def _get_object_location_with_visibility(self, sensor_id: str, object_info: Dict[str, str], bbox: Bbox) -> Tuple[Optional[List[float]], Optional[float]]:
        """
        Gets object location and visibility

        :param str sensor_id: sensor ID
        :param Dict[str,str] object_info: object info
        :param Bbox bbox: bounding box
        :return: object location and visibility
        :rtype: Tuple[Optional[List[float]],Optional[float]]
        """
        # Create homography array
        homography = None
        if (sensor_id in self.sensor_state_objects.keys()) and (self.sensor_state_objects[sensor_id].homography is not None):
            homography = self.sensor_state_objects[sensor_id].homography
        
        foot_pixel = None
        if (object_info is not None) and ("footLocation" in object_info.keys()):
            try:
                foot_tokens = object_info["footLocation"].split(",")
                foot_pixel = [float(foot_tokens[0]), float(foot_tokens[1])]
                del foot_tokens
            except ValueError:
                pass

        if foot_pixel is None:
            foot_pixel = [(bbox.leftX + bbox.rightX) / 2., bbox.bottomY]

        visibility = None
        if (object_info is not None) and ("visibility" in object_info.keys()):
            try:
                visibility = float(object_info["visibility"])
            except ValueError:
                visibility = 1.

        # Rectify bounding box based on calibration 
        if (self.people_height_estimator is not None) and (self.config.localization.rectifyBboxByCalibration):
            foot_pixel, visibility = self.people_height_estimator.rectify_bbox(foot_pixel, bbox, sensor_id)

        # Project the foot point to the ground plane using the homography
        location = None
        if (foot_pixel is not None) and (homography is not None):
            if (foot_pixel[0] > FOOT_PIXEL_MARGIN) or (foot_pixel[1] > FOOT_PIXEL_MARGIN):
                location = perspective_transform(foot_pixel, homography)
        del foot_pixel
        del homography
        return location, visibility
    
    def _create_behavior(self, frame_id: str, timestamp: datetime, behavior_id: str, sensor_id: str, object_id: str, object_type: str, bbox: Bbox, confidence: float) -> Behavior:
        """
        Creates a behavior object

        :param str frame_id: frame ID
        :param datetime timestamp: timestamp
        :param str behavior_id: behavior ID
        :param str sensor_id: sensor ID
        :param str object_id: object ID
        :param str object_type: object type
        :param Bbox bbox: bounding box
        :param float confidence: confidence
        :return: behavior object
        :rtype: Behavior
        """
        place = ""
        if sensor_id in self.sensor_state_objects.keys():
            place = self.sensor_state_objects[sensor_id].placeStr
        return Behavior(
            key=behavior_id, id=behavior_id, sensorId=sensor_id, objectId=object_id, objectType=object_type,
            timestamp=timestamp, end=timestamp, startFrame=frame_id, endFrame=frame_id,
            place=place, matchedSystemTimestamp=None, timestamps=[timestamp], frameIds=[frame_id],
            bboxes=[bbox], confidences=[confidence], locations=list(), locationMask=list(),
            embeddings=list(), embeddingMask=list(), roiMask=list()
        )

    def _update_behavior(self, behavior: Behavior, frame_id: str, timestamp: datetime, bbox: Bbox, confidence: float) -> None:
        """
        Updates a behavior object

        :param Behavior behavior: behavior object
        :param str frame_id: frame ID
        :param datetime timestamp: timestamp
        :param Bbox bbox: bounding box
        :param float confidence: confidence
        :return: None
        """
        if behavior.timestamp > timestamp:
            behavior.timestamp = timestamp
        if behavior.end < timestamp:
            behavior.end = timestamp
        if int(behavior.startFrame) > int(frame_id):
            behavior.startFrame = frame_id
        if int(behavior.endFrame) < int(frame_id):
            behavior.endFrame = frame_id
        behavior.timestamps.append(timestamp)
        behavior.frameIds.append(frame_id)
        behavior.bboxes.append(bbox)
        behavior.confidences.append(confidence)

    def _check_location_validity_and_update_behavior(self, behavior: Behavior, bbox: Bbox, confidence: float, location: Optional[List[float]]) -> None:
        """
        Checks location validity and updates behavior

        :param Behavior behavior: behavior object
        :param Bbox bbox: bounding box
        :param float confidence: confidence
        :param Optional[List[float]] location: location
        :return: None
        """
        sensor_id = behavior.sensorId
        if (sensor_id in self.sensor_state_objects.keys()) and \
            (not self._filter_by_confidence_and_bbox(confidence, bbox, sensor_id, self.config.preprocessing.locationConfidenceThresh, self.config.preprocessing.locationBboxAspectRatioThresh, self.config.preprocessing.locationBboxAreaThresh)) and \
            (not self._filter_by_bbox_touching_frame_bottom(bbox, sensor_id, self.config.preprocessing.locationBboxBottomGapThresh)):
            # When the location is None, it is possible to be localized in future micro-batches when the calibration is updated.
            behavior.locations.append(location)
            behavior.locationMask.append(True)
        else:
            behavior.locationMask.append(False)
    
    def _check_embedding_validity_and_update_behavior(self, behavior: Behavior, embedding: Optional[List[float]], bbox: Bbox, confidence: float, visibility: Optional[float]) -> None:
        """
        Checks location validity and updates behavior

        :param Behavior behavior: behavior object
        :param Optional[List[float]] embedding: embedding
        :param Bbox bbox: bounding box
        :param float confidence: confidence
        :param Optional[float] visibility: visibility
        :return: None
        """
        sensor_id = behavior.sensorId
        if (embedding is not None) and \
            (sensor_id in self.sensor_state_objects.keys()) and \
            (not self._filter_by_confidence_and_bbox(confidence, bbox, sensor_id, self.config.preprocessing.embeddingConfidenceThresh, self.config.preprocessing.embeddingBboxAspectRatioThresh, self.config.preprocessing.embeddingBboxAreaThresh)) and \
            (not self._filter_by_bbox_touching_frame_bottom(bbox, sensor_id, self.config.preprocessing.embeddingBboxBottomGapThresh)) and \
            ((visibility is None) or (visibility > self.config.preprocessing.embeddingVisibilityThresh)):
            behavior.embeddings.append(embedding)
            behavior.embeddingMask.append(True)
        else:
            behavior.embeddingMask.append(False)
    
    def _filter_by_regions_of_interest(self, behavior: Behavior, location: Optional[List[float]], roi_paths: List[mplPath.Path]) -> None:
        """
        Filters by regions of interest

        :param Behavior behavior: behavior object
        :param Optional[List[float]] location: location
        :param List[mplPath.Path] roi_paths: paths of regions of interest
        :return: None
        """
        is_in_roi = True
        if self.config.preprocessing.filterByRegionsOfInterest:
            if (location is None) or self._filter_object_by_rois(location, roi_paths):
                is_in_roi = False
        behavior.roiMask.append(is_in_roi)

    def _split_behaviors_in_list(self, behavior_list: List[Behavior]):
        """
        Splits behaviors in list

        :param List[Behavior] behavior_list: list of behaviors
        :return: list of behaviors after splitting
        :rtype: List[Behavior]
        """
        behaviors_after_splitting: List[Behavior] = list()
        for behavior in behavior_list:
            behaviors_after_splitting.extend(self._split_behavior(behavior))
        del behavior_list
        return behaviors_after_splitting

    def create_behaviors_from_protobuf_frames(self, protobuf_frames: List[Any]) -> List[Behavior]:
        """
        Creates behaviors from protobuf frames

        :param List[Any] protobuf_frames: protobuf frames
        :return: list of behaviors
        :rtype: List[Behavior]
        ::

            behaviors = preprocessor.create_behaviors_from_protobuf_frames(protobuf_frames)
        """
        behaviors: Dict[str, Behavior] = dict()
        timestamp_current = datetime.utcnow().replace(tzinfo=timezone.utc)
        roi_paths_dict: Dict[str, List[mplPath.Path]] = dict()

        for frame in protobuf_frames:
            timestamp_ms = int((frame.timestamp.seconds + frame.timestamp.nanos * (10 ** -9))*1000)
            timestamp_str = f'{datetime.utcfromtimestamp(timestamp_ms / 1000).isoformat(timespec="milliseconds")}Z'
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z","+00:00"))

            # Check if the current frame is within the given timestamp threshold
            if self.config.preprocessing.timestampThreshMin is not None:
                if timestamp_current > (timestamp + timedelta(seconds=int(self.config.preprocessing.timestampThreshMin * 60))):
                    continue

            sensor_id = frame.sensorId
            frame_id = frame.id

            # Create paths of regions of interest
            if sensor_id not in roi_paths_dict:
                roi_paths_dict[sensor_id] = self._get_roi_paths(sensor_id)

            for object_instance in frame.objects:
                bbox = Bbox(
                            leftX=object_instance.bbox.leftX,
                            topY=object_instance.bbox.topY,
                            rightX=object_instance.bbox.rightX,
                            bottomY=object_instance.bbox.bottomY
                        )
                location, visibility = self._get_object_location_with_visibility(sensor_id, object_instance.info, bbox)

                object_id = object_instance.id
                behavior_id = sensor_id + " #-# " + object_id
                object_type = object_instance.type

                confidence = object_instance.confidence
                # Handle empty confidence
                if confidence == 0.:
                    confidence = 1.

                embedding = None
                if hasattr(object_instance, "embedding") and hasattr(object_instance.embedding, "vector"):
                    embedding = normalize_vector(object_instance.embedding.vector)
                    if embedding is not None:
                        embedding = embedding.tolist()

                if behavior_id not in behaviors.keys():
                    behaviors[behavior_id] = self._create_behavior(frame_id, timestamp, behavior_id, sensor_id, object_id, object_type, bbox, confidence) 
                else:
                    if behaviors[behavior_id].objectType != object_type:
                        logging.error(f"ERROR: The object types do not match -- {behaviors[behavior_id].objectType} != {object_type}.")
                        exit(1)
                    self._update_behavior(behaviors[behavior_id], frame_id, timestamp, bbox, confidence)

                # Check validity of the location
                self._check_location_validity_and_update_behavior(behaviors[behavior_id], bbox, confidence, location)

                # Check validity of the embedding
                self._check_embedding_validity_and_update_behavior(behaviors[behavior_id], embedding, bbox, confidence, visibility)

                # Filter by regions of interest
                self._filter_by_regions_of_interest(behaviors[behavior_id], location, roi_paths_dict[sensor_id])

        # Split behaviors based on the gap of timestamps
        behavior_list = self._split_behaviors_in_list(list(behaviors.values()))

        # Apply RoI masks
        behavior_list_to_return = list(filter(None, map(self._apply_roi_mask, behavior_list)))
        del behavior_list

        # Sum embeddings of behaviors
        behavior_list_to_return = self.sum_embeddings(behavior_list_to_return)

        # Set behavior keys
        for i in range(len(behavior_list_to_return)):
            behavior_list_to_return[i].key += " #-# " + \
                behavior_list_to_return[i].timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")

        logging.debug(f"Created {len(behavior_list_to_return)} behavior(s) from {len(protobuf_frames)} frame(s)")

        return behavior_list_to_return

    def create_behaviors_from_frames(self, frames: List[Frame]) -> List[Behavior]:
        """
        Creates behaviors from frames

        :param List[Frame] frames: list of frames
        :return: list of behaviors
        :rtype: List[Behavior]
        ::

            behaviors = preprocessor.create_behaviors_from_frames(frames)
        """
        behaviors: Dict[str, Behavior] = dict()
        timestamp_current = datetime.utcnow().replace(tzinfo=timezone.utc)
        roi_paths_dict: Dict[str, List[mplPath.Path]] = dict()
        selected_sensor_ids = set(self.config.io.selectedSensorIds)
        
        for frame in frames:
            
            sensor_id = frame.sensorId

            if (len(selected_sensor_ids) > 0) and (sensor_id not in selected_sensor_ids):
                continue

            frame_id = frame.id
            timestamp = frame.timestamp

            # Check if the current frame is within the given timestamp threshold
            if self.config.preprocessing.timestampThreshMin is not None:
                if timestamp_current > (timestamp + timedelta(seconds=int(self.config.preprocessing.timestampThreshMin * 60))):
                    continue

            # Create RoI paths
            if sensor_id not in roi_paths_dict:
                roi_paths_dict[sensor_id] = self._get_roi_paths(sensor_id)

            for object_instance in frame.objects:
                
                bbox = object_instance.bbox

                location, visibility = self._get_object_location_with_visibility(sensor_id, object_instance.info, bbox)
                
                object_id = object_instance.id
                behavior_id = sensor_id + " #-# " + object_id
                object_type = object_instance.type
                confidence = object_instance.confidence
                embedding = object_instance.embedding

                if behavior_id not in behaviors.keys():
                    behaviors[behavior_id] = self._create_behavior(frame_id, timestamp, behavior_id, sensor_id, object_id, object_type, bbox, confidence) 
                else:
                    if behaviors[behavior_id].objectType != object_type:
                        logging.error(f"ERROR: The object types do not match -- {behaviors[behavior_id].objectType} != {object_type}.")
                        exit(1)
                    self._update_behavior(behaviors[behavior_id], frame_id, timestamp, bbox, confidence)

                # Check validity of the location
                self._check_location_validity_and_update_behavior(behaviors[behavior_id], bbox, confidence, location)
                
                # Check validity of the embedding
                self._check_embedding_validity_and_update_behavior(behaviors[behavior_id], embedding, bbox, confidence, visibility)
                
                # Filter by regions of interest
                self._filter_by_regions_of_interest(behaviors[behavior_id], location, roi_paths_dict[sensor_id])

        # Split behaviors based on the gap of timestamps
        behavior_list = self._split_behaviors_in_list(list(behaviors.values()))

        # Apply RoI masks
        behavior_list_to_return = list(filter(None, map(self._apply_roi_mask, behavior_list)))
        del behavior_list

        # Sum embeddings of behaviors
        behavior_list_to_return = self.sum_embeddings(behavior_list_to_return)

        # Set behavior keys
        for i in range(len(behavior_list_to_return)):
            behavior_list_to_return[i].key += " #-# " + \
                behavior_list_to_return[i].timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")

        logging.debug(f"Created {len(behavior_list_to_return)} behavior(s) from {len(frames)} frame(s)")

        return behavior_list_to_return

    def sample_locations(self, behavior: Behavior) -> Behavior:
        """
        Sample locations of a long behavior

        :param Behavior behavior: behavior object
        :return: behavior after the locations being sampled
        :rtype: Behavior
        ::

            sampled_behavior = preprocessor.sample_locations(behavior)
        """
        while len(behavior.timestamps) > self.config.preprocessing.behaviorNumLocationsMax:
            behavior.timestamps = behavior.timestamps[::2]
            behavior.frameIds = behavior.frameIds[::2]
            behavior.bboxes = behavior.bboxes[::2]
            behavior.confidences = behavior.confidences[::2]

            locations: List[List[float]] = list()
            idx_location = 0
            for i in range(len(behavior.locationMask)):
                if behavior.locationMask[i]:
                    if i % 2 == 0:
                        locations.append(behavior.locations[idx_location])
                    idx_location += 1
            behavior.locations = locations
            behavior.locationMask = behavior.locationMask[::2]

        return behavior

    def filter_by_confidence(self, behavior: Behavior) -> bool:
        """
        Filters behavior by mean detection confidence

        :param Behavior behavior: behavior object
        :return: decision to filter the behavior
        :rtype: bool
        ::

            filtered_behavior = preprocessor.filter_by_confidence(behavior)
        """
        confidences: List[float] = list()

        for confidence in behavior.confidences:
            if confidence >= 0:
                confidences.append(confidence)

        if len(confidences) == 0:
            return True

        behavior_confidence = sum(confidences) / len(confidences)

        if behavior_confidence < self.config.preprocessing.behaviorConfidenceThresh:
            return True

        return False

    def filter_by_bbox(self, behavior: Behavior) -> bool:
        """
        Filters behavior by bounding boxes

        :param Behavior behavior: behavior object
        :return: decision to filter the behavior
        :rtype: bool
        ::

            filtered_behavior = preprocessor.filter_by_bbox(behavior)
        """
        bbox_areas: List[float] = list()
        bbox_aspect_ratios: List[float] = list()

        for bbox in behavior.bboxes:
            bbox_areas.append(calculate_bbox_area(bbox))
            bbox_aspect_ratios.append(calculate_bbox_aspect_ratio(bbox))

        if (len(bbox_areas) == 0) or (len(bbox_aspect_ratios) == 0):
            return True

        if (sum(bbox_aspect_ratios) / len(bbox_aspect_ratios)) >= self.config.preprocessing.behaviorBboxAspectRatioThresh:
            return True

        if (behavior.sensorId in self.sensor_state_objects.keys()):
            frame_width = self.sensor_state_objects[behavior.sensorId].frameWidth
            frame_height = self.sensor_state_objects[behavior.sensorId].frameHeight
            if (frame_width is not None) and (frame_height is not None):
                if (sum(bbox_areas) / len(bbox_areas)) <= (frame_width * frame_height * self.config.preprocessing.behaviorBboxAreaThresh):
                    return True

        return False

    def filter_by_behavior_length(self, behavior: Behavior) -> bool:
        """
        Filters behavior by behavior length in time

        :param Behavior behavior: behavior object
        :return: decision to filter the behavior
        :rtype: bool
        ::

            filtered_behavior = preprocessor.filter_by_behavior_length(behavior)
        """
        # Always preserve long behaviors
        if (behavior.end - behavior.timestamp).total_seconds() >= self.config.preprocessing.behaviorLengthThreshSec:
            return False

        # Filter a short behavior if the threshold of finishing time is null
        if self.config.preprocessing.shortBehaviorFinishThreshSec is None:
            return True

        # Check if the short behavior has finished by comparing with the current time
        timestamp_current = datetime.utcnow().replace(tzinfo=timezone.utc)
        if timestamp_current < (behavior.end + timedelta(seconds=self.config.preprocessing.shortBehaviorFinishThreshSec)):
            return True

        return False

    def filter(self, behaviors: List[Behavior]) -> List[Behavior]:
        """
        Filters behaviors

        :param List[Behavior] behaviors: list of behaviors
        :return: filtered list of behaviors
        :rtype: List[Behavior]
        ::

            filtered_behaviors = preprocessor.filter(behaviors)
        """
        logging.info(f"Filtering behaviors...")
        filtered_behaviors: List[Behavior] = list()
        num_behaviors_filtered_by_confidence = 0
        num_behaviors_filtered_by_bbox = 0
        num_behaviors_filtered_by_behavior_length = 0
        num_behaviors_empty_embedding = 0

        for behavior in behaviors:
            # Filter behaviors based on confidence
            if self.filter_by_confidence(behavior):
                num_behaviors_filtered_by_confidence += 1
                continue

            # Filter behaviors based on bbox
            if self.filter_by_bbox(behavior):
                num_behaviors_filtered_by_bbox += 1
                continue

            # Filter behaviors based on behavior length
            if self.filter_by_behavior_length(behavior):
                num_behaviors_filtered_by_behavior_length += 1
                continue

            # Filter behaviors when it has empty embedding
            if len(behavior.embeddings) == 0:
                num_behaviors_empty_embedding += 1
                continue

            filtered_behaviors.append(behavior)

        logging.info(f"No. behaviors filtered by confidence: {num_behaviors_filtered_by_confidence}")
        logging.info(f"No. behaviors filtered by bbox: {num_behaviors_filtered_by_bbox}")
        logging.info(f"No. behaviors filtered by behavior length: {num_behaviors_filtered_by_behavior_length}")
        logging.info(f"No. behaviors with empty embedding: {num_behaviors_empty_embedding}")
        logging.info(f"No. behaviors after filtering: {len(filtered_behaviors)}")

        return filtered_behaviors

    def sum_embeddings(self, behaviors: List[Behavior]) -> List[Behavior]:
        """
        Sums embeddings of behaviors

        :param List[Behavior] behaviors: list of behaviors
        :return: behaviors with summed embeddings
        :rtype: List[Behavior]
        ::

            behaviors = preprocessor.sum_embeddings(behaviors)
        """
        for i in range(len(behaviors)):
            if len(behaviors[i].embeddings) > 1:
                behaviors[i].embeddings = [np.sum(np.array(behaviors[i].embeddings), axis=0).tolist()]
            behaviors[i].embeddingMask = None

        return behaviors

    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Normalizes the embeddings of a behavior

        :param List[List[float]] embeddings: list of embeddings
        :return: normalized embeddings
        :rtype: List[List[float]]
        ::

            normalized_embeddings = preprocessor.normalize_embeddings(embeddings)
        """
        normalized_embeddings: List[List[float]] = list()
        for embedding in embeddings:
            embedding = normalize_vector(embedding)
            if embedding is not None:
                normalized_embeddings.append(embedding.tolist())
            else:
                logging.error(f"ERROR: The embedding is None, caused by zero norm.")
                exit(1)
        return normalized_embeddings

    def group_behaviors_by_places(self, behaviors: List[Behavior]) -> Dict[str, List[Behavior]]:
        """
        Groups behaviors by places

        :param List[Behavior] behaviors: list of behaviors
        :return: map from places to behaviors
        :rtype: Dict[str, List[Behavior]]
        ::

            map_place_to_behaviors = preprocessor.group_behaviors_by_places(behaviors)
        """
        map_place_to_behaviors: Dict[str, List[Behavior]] = defaultdict(list)
        for behavior in behaviors:
            map_place_to_behaviors[behavior.place].append(behavior.copy())
        return map_place_to_behaviors

    def stitch_behaviors(self, behaviors: List[Behavior]) -> List[Behavior]:
        """
        Stitches behaviors sharing the same behavior IDs

        :param List[Behavior]: list of behaviors
        :return: behaviors after stitching
        :rtype: List[Behavior]
        ::

            stitched_behaviors = preprocessor.stitch_behaviors(behaviors)
        """
        # Group behaviors by behavior IDs
        map_behavior_id_to_behaviors: Dict[str, List[Behavior]] = defaultdict(list)
        for behavior in behaviors:
            map_behavior_id_to_behaviors[behavior.id].append(behavior.copy())

        # Stitch behaviors with the same IDs
        stitched_behaviors: List[Behavior] = list()
        for behavior_id in map_behavior_id_to_behaviors.keys():
            # Append behavior directly if there is no duplication
            if len(map_behavior_id_to_behaviors[behavior_id]) == 1:
                stitched_behaviors.append(map_behavior_id_to_behaviors[behavior_id][0])
                continue

            sensor_id = map_behavior_id_to_behaviors[behavior_id][0].sensorId
            object_id = map_behavior_id_to_behaviors[behavior_id][0].objectId
            object_type = map_behavior_id_to_behaviors[behavior_id][0].objectType
            place = map_behavior_id_to_behaviors[behavior_id][0].place
            timestamps: List[datetime] = list()
            frame_ids: List[str] = list()
            bboxes: List[Bbox] = list()
            confidences: List[float] = list()
            location_mask: List[bool] = list()
            embeddings: List[List[float]] = list()
            roi_mask: List[bool] = list()

            # Merge timestamps
            for behavior in map_behavior_id_to_behaviors[behavior_id]:
                timestamps.extend(behavior.timestamps)

            # Create a mapping from old to new indices
            map_timestamps_from_old_indices_to_new_indices: Dict[int, int] = \
                {old: new for new, old in enumerate(range(len(timestamps)))}

            # Initialize temporary lists for locations
            all_locations: List[List[float]] = list()

            # Initialize mappings for locations
            map_locations_from_old_indices_to_new_indices: Dict[int, int] = dict()

            # Merge and sort other lists
            location_counter = 0
            location_mask_counter = 0
            for behavior in map_behavior_id_to_behaviors[behavior_id]:
                frame_ids.extend(behavior.frameIds)
                bboxes.extend(behavior.bboxes)
                confidences.extend(behavior.confidences)
                all_locations.extend(behavior.locations)
                location_mask.extend(behavior.locationMask)
                embeddings.extend(behavior.embeddings)
                if behavior.roiMask is not None:
                    roi_mask.extend(behavior.roiMask)

                # Process location mask
                for mask in behavior.locationMask:
                    if mask:
                        map_locations_from_old_indices_to_new_indices[location_counter] = \
                            map_timestamps_from_old_indices_to_new_indices[location_mask_counter]
                        location_counter += 1
                    location_mask_counter += 1

            # Sort based on timestamps
            frame_ids = [frame_ids[map_timestamps_from_old_indices_to_new_indices[i]] for i in range(len(frame_ids))]
            bboxes = [bboxes[map_timestamps_from_old_indices_to_new_indices[i]] for i in range(len(bboxes))]
            confidences = [confidences[map_timestamps_from_old_indices_to_new_indices[i]] for i in range(len(confidences))]
            location_mask = [location_mask[map_timestamps_from_old_indices_to_new_indices[i]] for i in range(len(location_mask))]
            if len(roi_mask) == len(timestamps):
                roi_mask = [roi_mask[map_timestamps_from_old_indices_to_new_indices[i]] for i in range(len(roi_mask))]
            else:
                roi_mask = None
            del map_timestamps_from_old_indices_to_new_indices

            # Reorder locations
            sorted_locations = [None] * len(location_mask)

            for idx_old, idx_new in map_locations_from_old_indices_to_new_indices.items():
                sorted_locations[idx_new] = all_locations[idx_old]
            del all_locations
            del map_locations_from_old_indices_to_new_indices

            # Remove None values which represent masked out locations
            locations = [location for location in sorted_locations if location is not None]

            # Sum embeddings
            if len(embeddings) > 1:
                embeddings = [np.sum(np.array(embeddings), axis=0).tolist()]

            # Initialize the fields for the stitched behavior
            behavior_key = behavior_id + " #-# " + timestamps[0].isoformat(timespec="milliseconds").replace("+00:00", "Z")
            stitched_behavior = Behavior(
                key=behavior_key, id=behavior_id, sensorId=sensor_id, objectId=object_id, objectType=object_type,
                timestamp=timestamps[0], end=timestamps[-1], startFrame=frame_ids[0], endFrame=frame_ids[-1],
                place=place, matchedSystemTimestamp=None, timestamps=timestamps, frameIds=frame_ids,
                bboxes=bboxes, confidences=confidences, locations=locations, locationMask=location_mask,
                embeddings=embeddings, embeddingMask=None, roiMask=roi_mask
            )

            stitched_behaviors.append(stitched_behavior)

        del map_behavior_id_to_behaviors

        return stitched_behaviors

    def delay_behavior_timestamps(self, behaviors: List[Behavior]) -> List[Behavior]:
        """
        Delays timestamps of behaviors

        :param List[Behavior]: list of behaviors
        :return: behaviors after delaying timestamps of behaviors
        :rtype: List[Behavior]
        ::

            delayed_behaviors = preprocessor.delay_behavior_timestamps(behaviors)
        """
        # Compute max ending timestamp and set timestamp threshold
        max_end = None
        for behavior in behaviors:
            if (max_end is None) or (behavior.end > max_end):
                max_end = behavior.end
        timestamp_thresh = max_end - timedelta(milliseconds=self.config.streaming.mtmcPlusTimestampDelayMs)

        delayed_behaviors: List[Behavior] = list()
        for i in range(len(behaviors)):
            # Add the entire behavior if the ending timestamp is before the timestamp threshold
            if behaviors[i].end < timestamp_thresh:
                delayed_behaviors.append(behaviors[i])
                continue

            # Compute the index of timestamp limit
            idx_timestamp_limit = 0
            for j in range(len(behaviors[i].timestamps)):
                if behaviors[i].timestamps[j] > timestamp_thresh:
                    idx_timestamp_limit = j

            # Discard the behavior starts after the timestamp threshold
            if idx_timestamp_limit == 0:
                continue

            # Update lists and ending time for the behavior
            behaviors[i].timestamps = behaviors[i].timestamps[:idx_timestamp_limit]
            behaviors[i].end = behaviors[i].timestamps[-1]
            behaviors[i].frameIds = behaviors[i].frameIds[:idx_timestamp_limit]
            behaviors[i].endFrame = behaviors[i].frameIds[-1]
            behaviors[i].bboxes = behaviors[i].bboxes[:idx_timestamp_limit]
            behaviors[i].confidences = behaviors[i].confidences[:idx_timestamp_limit]
            behaviors[i].locationMask = behaviors[i].locationMask[:idx_timestamp_limit]
            behaviors[i].locations = behaviors[i].locations[:sum(behaviors[i].locationMask)]
            if behaviors[i].roiMask is not None:
                behaviors[i].roiMask = behaviors[i].roiMask[:idx_timestamp_limit]
            delayed_behaviors.append(behaviors[i])

        return delayed_behaviors

    def preprocess(self, frames: List[Frame]) -> List[Behavior]:
        """
        Preprocesses frames into behaviors and filters outliers

        :param List[Frame] frames: list of frames
        :return: list of behaviors
        :rtype: List[Behavior]
        ::

            behaviors = preprocessor.preprocess(frames)
        """
        # Create behaviors from frames
        behaviors = self.create_behaviors_from_frames(frames)

        # Sample locations of behaviors
        for i in range(len(behaviors)):
            behaviors[i] = self.sample_locations(behaviors[i])

        # Filter behaviors
        behaviors = self.filter(behaviors)

        # Normalizes embeddings of behaviors
        for i in range(len(behaviors)):
            behaviors[i].embeddings = self.normalize_embeddings(behaviors[i].embeddings)

        return behaviors
