# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import re
import logging
import pandas as pd
import mdx.mtmc.utils.trackeval as trackeval
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datetime import datetime

from mdx.mtmc.config import AppConfig
from mdx.mtmc.schema import Frame, Behavior, MTMCObject, SensorStateObject
from mdx.mtmc.core.data import perspective_transform
from mdx.mtmc.utils.io_utils import validate_file_path, load_csv_to_dataframe_from_file, write_dataframe_to_csv_file, \
                                    make_seq_maps_file, make_seq_ini_file, make_dir, remove_dir, move_files


class Evaluator:
    """
    Module to evaluate MTMC results

    :param dict config: configuration for the app
    ::

        evaluator = Evaluator(config)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self.selected_sensor_ids = set(config.io.selectedSensorIds)

    def _assign_duplicate_behaviors(self, mtmc_objects: List[MTMCObject]) -> List[MTMCObject]:
        """
        Detects duplicate behaviors in MTMC objects and chooses unique assignments based on matched system timestamps

        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :return: list of MTMC objects with unique assignments
        :rtype: List[MTMCObject]
        """
        behavior_matches: Dict[str, Tuple[str, datetime]] = dict()

        for mtmc_object in mtmc_objects:
            for behavior in mtmc_object.matched:
                behavior_key = behavior.key
                matched_system_timestamp = behavior.matchedSystemTimestamp

                # Choose the MTMC object matched at later end time
                if (matched_system_timestamp is None) or \
                    (behavior_key not in behavior_matches) or \
                    (behavior_matches[behavior_key][1] < matched_system_timestamp):
                    behavior_matches[behavior_key] = (mtmc_object.globalId, matched_system_timestamp)

        # Update the matched behaviors for each MTMC object
        updated_mtmc_objects: List[MTMCObject] = list()
        for mtmc_object in mtmc_objects:
            timestamp = None
            end = None
            matched: List[Behavior] = list()
            for behavior in mtmc_object.matched:
                behavior_key = behavior.key
                if behavior_matches[behavior_key][0] == mtmc_object.globalId:
                    if (timestamp is None) or (behavior.timestamp < timestamp):
                        timestamp = behavior.timestamp
                    if (end is None) or (behavior.end > end):
                        end = behavior.end
                    matched.append(behavior)
            mtmc_object.timestamp = timestamp
            mtmc_object.end = end
            mtmc_object.matched = matched
            updated_mtmc_objects.append(mtmc_object)

        return updated_mtmc_objects

    def _create_mot_pred(self, frames: List[Frame], mtmc_objects: List[MTMCObject], sensor_state_objects: Dict[str, SensorStateObject]) -> List[List[int]]:
        """
        Creates MOT prediction from MTMC objects and frames (for mapping timestamps)

        :param List[Frame] frames: list of frames
        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :return: MOT prediction for evaluation
        :rtype: List[List[int]]
        """
        # Create a map of MTMC objects
        mtmc_objects_map: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = defaultdict(list)
        sensor_ids = sorted(list(sensor_state_objects.keys()))
        for mtmc_object in mtmc_objects:
            global_id = int(mtmc_object.globalId)
            for matched_behavior in mtmc_object.matched:
                sensor_idx = sensor_ids.index(matched_behavior.sensorId) + 1
                object_id = int(matched_behavior.objectId)
                start_frame = int(matched_behavior.startFrame)
                end_frame = int(matched_behavior.endFrame)
                mtmc_objects_map[(sensor_idx, object_id)].append((global_id, start_frame, end_frame))

        # Create MOT prediction using information from frames
        mot_pred: List[List[int]] = list()
        for frame in frames:
            frame_id = int(frame.id)
            sensor_idx = sensor_ids.index(frame.sensorId) + 1

            # Create homography array
            homography = None
            if self.config.io.use3dEvaluation:
                if (frame.sensorId in sensor_state_objects.keys()) and \
                    (sensor_state_objects[frame.sensorId].homography is not None):
                    homography = sensor_state_objects[frame.sensorId].homography
                    if homography is None:
                        logging.error(f"ERROR: The homography for sensor ID {frame.sensorId} does NOT exist for 3D evaluation.")
                        exit(1)
    
            for object_instance in frame.objects:
                object_id = int(object_instance.id)
                mtmc_object_key = (sensor_idx, object_id)
                if mtmc_object_key not in mtmc_objects_map.keys():
                    continue
                bbox = object_instance.bbox
                foot_pixel = None
                if self.config.io.useFullBodyGroundTruth and (object_instance.info is not None) and ("footLocation" in object_instance.info.keys()):
                    foot_tokens = object_instance.info["footLocation"].split(",")
                    foot_pixel = [float(foot_tokens[0]), float(foot_tokens[1])]
                    del foot_tokens
                    if foot_pixel[0] < bbox.leftX:
                        bbox.leftX = foot_pixel[0]
                    if foot_pixel[0] > bbox.rightX:
                        bbox.rightX = foot_pixel[0]
                    if foot_pixel[1] < bbox.topY:
                        bbox.topY = foot_pixel[1]
                    if foot_pixel[1] > bbox.bottomY:
                        bbox.bottomY = foot_pixel[1]

                else:
                    foot_pixel = [(bbox.leftX + bbox.rightX) / 2., bbox.bottomY]

                x_world = None
                y_world = None
                if self.config.io.use3dEvaluation:
                    location = perspective_transform(foot_pixel, homography)
                    x_world = location[0]
                    y_world = location[1]

                bbox_x = int(bbox.leftX)
                bbox_y = int(bbox.topY)
                bbox_width = int(bbox.rightX - bbox_x + 1.)
                bbox_height = int(bbox.bottomY - bbox_y + 1.)

                for segment in mtmc_objects_map[mtmc_object_key]:
                    global_id, start_frame, end_frame = segment
                    if start_frame <= frame_id <= end_frame:
                        if self.config.io.use3dEvaluation:
                            mot_pred.append([sensor_idx, global_id, frame_id, bbox_x, bbox_y,
                                             bbox_width, bbox_height, x_world, y_world])
                        else:
                            mot_pred.append([sensor_idx, global_id, frame_id, bbox_x, bbox_y,
                                             bbox_width, bbox_height, -1, -1])
                        break
        del mtmc_objects_map
        del sensor_ids

        return mot_pred

    def _compute_2d_mot_metrics(self, frames: List[Frame], mtmc_objects: List[MTMCObject], sensor_state_objects: Dict[str, SensorStateObject]) -> None:
        """
        Computes 2D MOT metrics using frames & mtmc_objects. 
        The ground truth file will be obtained from the 'io.groundTruthPath' config.
        Plots will be generated if 'io.plotEvaluationGraphs' config is enabled.

        :param List[Frame] frames: list of frames
        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :return: None
        """

        # Get list of sensor IDs
        sensor_ids: Set[str] = set()
        for frame in frames:
            sensor_ids.add(frame.sensorId)
        sensor_ids = sorted(list(sensor_ids))

        # Create evaluater configs for trackeval lib
        default_eval_config = trackeval.eval.Evaluator.get_default_eval_config()
        default_eval_config["PRINT_CONFIG"] = False
        default_eval_config["USE_PARALLEL"] = True
        default_eval_config["LOG_ON_ERROR"] = None

        # Create dataset configs for trackeval lib
        default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()

        default_dataset_config["DO_PREPROC"] = False
        default_dataset_config["SPLIT_TO_EVAL"] = "all"
        default_dataset_config["GT_FOLDER"] = os.path.join(self.config.io.outputDirPath, "evaluation", "gt")
        default_dataset_config["TRACKERS_FOLDER"] = os.path.join(self.config.io.outputDirPath, "evaluation", "scores")
        default_dataset_config["PRINT_CONFIG"] = False

        # Make output directory for storing results
        make_dir(default_dataset_config["GT_FOLDER"])
        make_dir(default_dataset_config["TRACKERS_FOLDER"])

        # Set the metrics to obtain
        default_metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
        default_metrics_config["PRINT_CONFIG"] = False
        config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

        # Create MOT prediction
        mot_pred = self._create_mot_pred(frames, mtmc_objects, sensor_state_objects)
        # mot_pred = self._filter_multiple_behaviour(mot_pred)

        # Load ground truth and MOT prediction
        column_names = ["CameraId", "Id", "FrameId",
                        "X", "Y", "Width", "Height",
                        "Xworld", "Yworld"]
        ground_truth_path = self.config.io.groundTruthPath
        ground_truth_dataframe = load_csv_to_dataframe_from_file(ground_truth_path, column_names)
        mot_pred_dataframe = pd.DataFrame(mot_pred, columns=column_names)

        # Concatenate ground truth and MOT prediction from all sensors
        ground_truth_sensors = sorted(ground_truth_dataframe["CameraId"].drop_duplicates().tolist())

        # Filter sensor_ids which are not present in ground_truth_sensors
        sensor_ids = self._filter_sensors_by_ground_truth_ids(ground_truth_sensors, sensor_ids)

        # Create sequence maps file for evaluation
        seq_maps_file = os.path.join(default_dataset_config["GT_FOLDER"], "seqmaps")
        make_seq_maps_file(seq_maps_file, sensor_ids, default_dataset_config["BENCHMARK"], default_dataset_config["SPLIT_TO_EVAL"])

        mtmc_ground_truths: List[pd.DataFrame] = list()
        mtmc_mot_preds: List[pd.DataFrame] = list()
        frame_id_max = 0

        for sensor_gt_idx, sensor_id in zip(ground_truth_sensors, sensor_ids):
            if (len(self.selected_sensor_ids) > 0) and (sensor_id not in self.selected_sensor_ids):
                continue

            # Convert ground truth multi-camera dataframe to single camera in MOT format
            ground_truth = ground_truth_dataframe.query(f"CameraId == {sensor_gt_idx}")
            ground_truth = ground_truth[["FrameId", "Id", "X", "Y", "Width", "Height", "Xworld", "Yworld"]]
            ground_truth = ground_truth.drop_duplicates(subset=["FrameId", "Id"])

            # Align ground-truth and prediction frame IDs
            ground_truth["FrameId"] += self.config.io.groundTruthFrameIdOffset

            ground_truth["Conf"] = 1
            ground_truth["Xworld"] = -1
            ground_truth["Yworld"] = -1
            ground_truth["Zworld"] = -1
            ground_truth = ground_truth[["FrameId", "Id", "X", "Y", "Width", "Height", "Conf", "Xworld", "Yworld", "Zworld"]]

            # Remove logs for negative frame IDs
            ground_truth = ground_truth[ground_truth["FrameId"] >= 1]

            # Save single camera ground truth in MOT format as CSV
            mot_version = default_dataset_config["BENCHMARK"] + "-" + default_dataset_config["SPLIT_TO_EVAL"]
            gt_dir = os.path.join(default_dataset_config["GT_FOLDER"], mot_version)
            dir_name = os.path.join(gt_dir, sensor_id)
            gt_file_dir = os.path.join(gt_dir, sensor_id, "gt")
            gt_file_name = os.path.join(gt_file_dir, "gt.txt")
            make_dir(gt_file_dir)
            write_dataframe_to_csv_file(gt_file_name, ground_truth)

            frame_id_max_curr = ground_truth["FrameId"].max()
            ground_truth["FrameId"] += frame_id_max + 1
            mtmc_ground_truths.append(ground_truth)

            # Convert predicted multi-camera dataframe to MOT format
            mot_pred = mot_pred_dataframe.query(f"CameraId == {sensor_gt_idx}")
            mot_pred = mot_pred[["FrameId", "Id", "X", "Y", "Width", "Height"]]
            mot_pred = mot_pred.drop_duplicates(subset=["FrameId", "Id"])

            # Remove logs for negative frame ids
            mot_pred = mot_pred[mot_pred["FrameId"] >= 1]

            mot_pred["Conf"] = 1
            mot_pred["Xworld"] = -1
            mot_pred["Yworld"] = -1
            mot_pred["Zworld"] = -1
            mot_pred = mot_pred[["FrameId", "Id", "X", "Y", "Width", "Height", "Conf", "Xworld", "Yworld", "Zworld"]]
            
            # Save single camera prediction in MOT format as CSV
            mot_file_dir = os.path.join(default_dataset_config["TRACKERS_FOLDER"], mot_version, "data", "data")
            make_dir(mot_file_dir)
            tracker_file_name = str(sensor_id) + ".txt"
            mot_file_name = os.path.join(mot_file_dir, tracker_file_name)
            write_dataframe_to_csv_file(mot_file_name, mot_pred)

            frame_id_max_curr = max(frame_id_max_curr, mot_pred["FrameId"].max())
            mot_pred["FrameId"] += frame_id_max + 1
            mtmc_mot_preds.append(mot_pred)

            # Make sequence ini file for trackeval library
            if np.isnan(mot_pred["FrameId"].max()):
                last_frame_id = ground_truth["FrameId"].max()
            elif np.isnan(ground_truth["FrameId"].max()):
                last_frame_id = mot_pred["FrameId"].max()
            else:
                last_frame_id = max(mot_pred["FrameId"].max(), ground_truth["FrameId"].max())
            make_seq_ini_file(dir_name, camera=sensor_id, seq_length=last_frame_id)          

            frame_id_max += frame_id_max_curr

        # Write ground truth file with all sensors as csv
        dir_name = os.path.join(gt_dir, "MTMC")
        gt_file_dir = os.path.join(gt_dir, "MTMC", "gt")
        gt_file_name = os.path.join(gt_file_dir, "gt.txt")
        make_dir(gt_file_dir)
        write_dataframe_to_csv_file(gt_file_name, pd.concat(mtmc_ground_truths))

        # Write prediction file with all sensors as csv
        tracker_file_name = "MTMC" + ".txt"
        mot_file_name = os.path.join(mot_file_dir, tracker_file_name)
        make_dir(mot_file_dir)
        write_dataframe_to_csv_file(mot_file_name, pd.concat(mtmc_mot_preds))
        make_seq_ini_file(dir_name, camera="MTMC", seq_length=frame_id_max+1)          

        # Evaluate ground truth & prediction to get all exhaustive metrics
        evaluator = trackeval.eval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        temp_metrics_list = [trackeval.metrics.HOTA, trackeval.metrics.CLEAR,trackeval.metrics.Identity]

        metrics_list = []
        for metric in temp_metrics_list:
            if metric.get_name() in metrics_config["METRICS"]:
                metrics_list.append(metric(metrics_config))

        if len(metrics_list) == 0:
            raise Exception("No metrics selected for evaluation.")
        results = evaluator.evaluate(dataset_list, metrics_list)

        # Extracting the largest global id assigned in the last batch
        max_global_id = -1
        last_batch_id = -1

        for mtmc_object in mtmc_objects:
            current_batch_id = int(mtmc_object.batchId)
            current_global_id = int(mtmc_object.globalId)

            if current_batch_id > last_batch_id:
                last_batch_id = current_batch_id
                max_global_id = current_global_id
            elif current_batch_id == last_batch_id:
                max_global_id = max(max_global_id, current_global_id)
        
        # Extract the total ground-truth global IDs
        gt_global_id_count = results[0]["MotChallenge2DBox"]["data"]["MTMC"]["pedestrian"]["Count"]["GT_IDs"]

        logging.info(f"The max global ID assigned in the final batch is: {max_global_id}")
        logging.info(f"The total count of global IDs in ground truth is: {gt_global_id_count}")
        
        # Generate plots for analysis between detection, localization & association.
        if self.config.io.plotEvaluationGraphs:
            self.generate_plots(default_dataset_config, mot_version)

        # Clean up result directory
        if not self.config.io.enableDebug:
            self._remove_result_dirs(default_dataset_config, mot_version)

        return


    def _compute_3d_mot_metrics(self, frames: List[Frame], mtmc_objects: List[MTMCObject], sensor_state_objects: Dict[str, SensorStateObject]) -> None:
        """
        Computes 3D MOT metrics using frames & mtmc_objects. 
        The ground truth file will be obtained from the 'io.groundTruthPath' config.
        Plots will be generated if 'io.plotEvaluationGraphs' config is enabled.

        :param List[Frame] frames: list of frames
        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :return: None
        """

        # Load ground truth and prediction files in dataframe
        column_names = ["CameraId", "Id", "FrameId", "X", "Y", "Width", "Height", "Xworld", "Yworld"]
        
        ground_truth_path = self.config.io.groundTruthPath
        ground_truth_dataframe = load_csv_to_dataframe_from_file(ground_truth_path, column_names)
        
        mot_pred = self._create_mot_pred(frames, mtmc_objects, sensor_state_objects)
        mot_pred_dataframe = pd.DataFrame(mot_pred, columns=column_names)

        # Create evaluater configs for trackeval lib
        default_eval_config = trackeval.eval.Evaluator.get_default_eval_config()
        default_eval_config["PRINT_CONFIG"] = False
        default_eval_config["USE_PARALLEL"] = True
        default_eval_config["LOG_ON_ERROR"] = None

        # Create dataset configs for trackeval lib
        default_dataset_config = trackeval.datasets.MotChallenge3DLocation.get_default_dataset_config()
        default_dataset_config["DO_PREPROC"] = False
        default_dataset_config["SPLIT_TO_EVAL"] = "all"
        default_dataset_config["GT_FOLDER"] = os.path.join(self.config.io.outputDirPath, "evaluation", "gt")
        default_dataset_config["TRACKERS_FOLDER"] = os.path.join(self.config.io.outputDirPath, "evaluation", "scores")
        default_dataset_config["PRINT_CONFIG"] = False

        # Make output directory for storing results
        make_dir(default_dataset_config["GT_FOLDER"])
        make_dir(default_dataset_config["TRACKERS_FOLDER"])

        # Create sequence maps file for evaluation
        seq_maps_file = os.path.join(default_dataset_config["GT_FOLDER"], "seqmaps")
        make_seq_maps_file(seq_maps_file, set(), default_dataset_config["BENCHMARK"], default_dataset_config["SPLIT_TO_EVAL"])

        # Set the metrics to obtain
        default_metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
        default_metrics_config["PRINT_CONFIG"] = False
        config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

        # Convert ground truth multi-camera dataframe to single camera in MOT format
        ground_truth_dataframe_per_scene = ground_truth_dataframe.drop_duplicates(subset=["FrameId", "Id"])
        ground_truth_dataframe_per_scene = ground_truth_dataframe_per_scene[["FrameId", "Id", "X", "Y", "Width", "Height", "Xworld", "Yworld"]]
        ground_truth_dataframe_per_scene = ground_truth_dataframe_per_scene.sort_values(by="FrameId")

        # Align ground-truth and prediction frame IDs
        ground_truth_dataframe_per_scene["FrameId"] += self.config.io.groundTruthFrameIdOffset

        # Set other defaults
        ground_truth_dataframe_per_scene["Conf"] = 1
        ground_truth_dataframe_per_scene["Zworld"] = -1
        ground_truth_dataframe_per_scene = ground_truth_dataframe_per_scene[["FrameId", "Id", "X", "Y", "Width", "Height", "Conf", "Xworld", "Yworld", "Zworld"]]

        # Remove logs for negative frame ids
        ground_truth_dataframe_per_scene = ground_truth_dataframe_per_scene[ground_truth_dataframe_per_scene["FrameId"] >= 1]

        # Save single camera ground truth in MOT format as CSV
        mot_version = default_dataset_config["BENCHMARK"] + "-" + default_dataset_config["SPLIT_TO_EVAL"]
        gt_dir = os.path.join(default_dataset_config["GT_FOLDER"], mot_version)
        dir_name = os.path.join(gt_dir, "MTMC")
        gt_file_dir = os.path.join(gt_dir, "MTMC", "gt")
        gt_file_name = os.path.join(gt_file_dir, "gt.txt")
        make_dir(gt_file_dir)
        write_dataframe_to_csv_file(gt_file_name, ground_truth_dataframe_per_scene)

        # Convert predicted multi-camera dataframe to MOT format
        mot_pred_dataframe_per_scene = mot_pred_dataframe.drop_duplicates(subset=["FrameId", "Id"])
        mot_pred_dataframe_per_scene = mot_pred_dataframe_per_scene[["FrameId", "Id", "X", "Y", "Width", "Height", "Xworld", "Yworld"]]
        mot_pred_dataframe_per_scene = mot_pred_dataframe_per_scene.sort_values(by="FrameId")

        # Remove logs for negative frame ids
        mot_pred_dataframe_per_scene = mot_pred_dataframe_per_scene[mot_pred_dataframe_per_scene["FrameId"] >= 1]

        # Set other defaults
        mot_pred_dataframe_per_scene["Conf"] = 1
        mot_pred_dataframe_per_scene["Zworld"] = -1
        mot_pred_dataframe_per_scene = mot_pred_dataframe_per_scene[["FrameId", "Id", "X", "Y", "Width", "Height", "Conf", "Xworld", "Yworld", "Zworld"]]
        
        # Save single camera prediction in MOT format as CSV
        mot_file_dir = os.path.join(default_dataset_config["TRACKERS_FOLDER"], mot_version, "data", "data")
        make_dir(mot_file_dir)
        tracker_file_name = "MTMC.txt"
        mot_file_name = os.path.join(mot_file_dir, tracker_file_name)
        write_dataframe_to_csv_file(mot_file_name, mot_pred_dataframe_per_scene)

        # Make sequence ini file for trackeval library
        if np.isnan(mot_pred_dataframe_per_scene["FrameId"].max()):
            last_frame_id = ground_truth_dataframe_per_scene["FrameId"].max()
        elif np.isnan(ground_truth_dataframe_per_scene["FrameId"].max()):
            last_frame_id = mot_pred_dataframe_per_scene["FrameId"].max()
        else:
            last_frame_id = max(mot_pred_dataframe_per_scene["FrameId"].max(), ground_truth_dataframe_per_scene["FrameId"].max())
        make_seq_ini_file(dir_name, camera="MTMC", seq_length=last_frame_id)     

        # Evaluate ground truth & prediction to get all exhaustive metrics
        evaluator = trackeval.eval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge3DLocation(dataset_config)]
        temp_metrics_list = [trackeval.metrics.HOTA, trackeval.metrics.CLEAR,trackeval.metrics.Identity]

        metrics_list = []
        for metric in temp_metrics_list:
            if metric.get_name() in metrics_config["METRICS"]:
                metrics_list.append(metric(metrics_config))

        results = evaluator.evaluate(dataset_list, metrics_list)

        # Extracting the largest global id assigned in the last batch
        max_global_id = -1
        last_batch_id = -1

        for mtmc_object in mtmc_objects:
            current_batch_id = int(mtmc_object.batchId)
            current_global_id = int(mtmc_object.globalId)

            if current_batch_id > last_batch_id:
                last_batch_id = current_batch_id
                max_global_id = current_global_id
            elif current_batch_id == last_batch_id:
                max_global_id = max(max_global_id, current_global_id)
        
        # Extract the total ground-truth global IDs
        gt_global_id_count = results[0]["MotChallenge3DLocation"]["data"]["MTMC"]["pedestrian"]["Count"]["GT_IDs"]

        logging.info(f"The max global ID assigned in the final batch is: {max_global_id}")
        logging.info(f"The total count of global IDs in ground truth is: {gt_global_id_count}")
        
        # Generate plots for analysis between detection, localization & association
        if self.config.io.plotEvaluationGraphs:
            self.generate_plots(default_dataset_config, mot_version)

        # Clean up result directory
        if not self.config.io.enableDebug:
            self._remove_result_dirs(default_dataset_config, mot_version)

        return

    def _filter_sensors_by_ground_truth_ids(self, ground_truth_sensors: List[float], sensor_ids: List[str]):
        """
        Filters a list of sensor IDs, retaining only those whose numeric part matches any of the ground truth IDs.

        :param List[float] ground_truth_ids: Set of IDs considered as ground truth for matching against the numeric part of sensor IDs.
        :param List[str] sensor_ids: List of sensor IDs to be filtered.

        :return List[str]: Filtered list of sensor IDs based on ground truth matching.

        """
        ground_truth_sensors = [int(num) for num in ground_truth_sensors]

        filtered_sensor_ids = []
        for sensor_id in sensor_ids:
            sensor_id_in_gt = self._extract_sensor_number(sensor_id)
            if sensor_id_in_gt in ground_truth_sensors:
                filtered_sensor_ids.append(sensor_id)
            else:
                logging.info(f'Skipped evaluation for sensor ID {sensor_id} as its ID {sensor_id_in_gt} could not be found in ground truth file.')
        return filtered_sensor_ids

    def _extract_sensor_number(self, sensor_id: str):
        """
        Extracts the numeric part from a sensor ID.

        :param str sensor_id: Sensor ID from which to extract the number.

        :return int or None: The extracted number as an integer, or None if no number is found.
        """
        match = re.search(r'\d+', sensor_id)
        return int(match.group()) if match else None

    def _generate_plots(self, default_dataset_config, mot_version):
        """
        Generates and organizes evaluation plots for the specified MOT version and dataset configuration. 
        It creates plots comparing tracker performances on defined object classes (e.g., pedestrian) and 
        saves them in an organized structure within the output directory.

        :param default_dataset_config: Configuration containing paths for the trackers folder and other relevant settings.
        :param mot_version: The version of the MOT challenge format for which plots are generated.
        :return: None
        """

        plots_folder = os.path.join(self.config.io.outputDirPath, "evaluation", "plots")
        make_dir(plots_folder)
        trackers_folder = default_dataset_config["TRACKERS_FOLDER"]
        classes = ["pedestrian"]
        data_folder = os.path.join(trackers_folder, mot_version)
        trackers = os.listdir(validate_file_path(data_folder))
        for cls in classes:
            trackeval.plotting.plot_compare_trackers(data_folder, trackers, cls, plots_folder)
        plot_folder_with_class = os.path.join(plots_folder, cls)
        move_files(source_dir=plot_folder_with_class, destination_dir=plots_folder)
        remove_dir(plot_folder_with_class)
        return

    def _remove_result_dirs(self, default_dataset_config, mot_version):
        """
        Cleans up intermediate directories and files created during the MOT evaluation process. 
        It removes ground truth directories, MOT data directories, and moves result files to a specified scores folder.

        :param default_dataset_config: Configuration containing paths for ground truth, trackers, and scores directories.
        :param mot_version: The version of the MOT challenge format being used.
        :return: None
        """

        # Remove all intermediate ground truth files created for evaluation
        remove_dir(default_dataset_config["GT_FOLDER"])

        # Remove all intermediate MOT files created during evaluation
        mot_data_file_dir = os.path.join(default_dataset_config["TRACKERS_FOLDER"], mot_version, "data", "data")
        remove_dir(mot_data_file_dir)

        # Move all the results to evaluation/scores folder
        mot_root_dir = os.path.join(default_dataset_config["TRACKERS_FOLDER"], mot_version, "data")
        move_files(source_dir=mot_root_dir, destination_dir=default_dataset_config["TRACKERS_FOLDER"])

        # Remove all MOT & class folders
        mot_dir = os.path.join(default_dataset_config["TRACKERS_FOLDER"], mot_version)
        remove_dir(mot_dir)
        return

    def evaluate(self, frames: List[Frame], mtmc_objects: List[MTMCObject], sensor_state_objects: Dict[str, SensorStateObject]) -> None:
        """
        Evaluates MTMC results

        :param List[Frame] frames: list of frames
        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :rtype: None
        ::

            evaluator.evaluate(frames, mtmc_objects, sensor_state_objects)
        """
        # Choose unique assignments for duplicate behaviors
        mtmc_objects = self._assign_duplicate_behaviors(mtmc_objects)

        # Compute MOT metrics
        if self.config.io.use3dEvaluation:
            self._compute_3d_mot_metrics(frames, mtmc_objects, sensor_state_objects)
        else:
            self._compute_2d_mot_metrics(frames, mtmc_objects, sensor_state_objects)
        return
