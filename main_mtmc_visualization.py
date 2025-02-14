# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import argparse
import logging
from multiprocessing import Pool
from itertools import repeat

from mdx.mtmc.schema import Frame, Behavior, MTMCObject
from mdx.mtmc.config import VizMtmcConfig
from mdx.mtmc.viz.frames import VizFrames
from mdx.mtmc.viz.behaviors import VizBehaviors
from mdx.mtmc.viz.mtmc_objects_in_grid import VizMTMCObjectsInGrid
from mdx.mtmc.viz.mtmc_objects_in_sequence import VizMTMCObjectsInSequence
from mdx.mtmc.viz.mtmc_objects_in_topview import VizMTMCObjectsInTopview
from mdx.mtmc.viz.ground_truth_locations import VizGroundTruthLocations
from mdx.mtmc.utils.io_utils import ValidateFile, validate_file_path, sanitize_string, \
    load_json_from_file, load_json_from_file_line_by_line, load_csv_from_file, make_clean_dir
from mdx.mtmc.utils.viz_mtmc_utils import hash_frames, hash_behaviors, hash_mtmc_objects, \
    hash_ground_truth_bboxes, hash_ground_truth_locations

logging.basicConfig(format="%(asctime)s.%(msecs)03d - %(message)s",
                    datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)


class VisualizationApp:
    """
    Controller module for visualization

    :param str config_path: path to the visualization config file
    :param str calibration_path: path to the calibration file in JSON format
    ::

        visualization_app = VisualizationApp(config_path, calibration_path)
    """

    def __init__(self, config_path: str, calibration_path: str):
        # Make sure the config file exists
        valid_config_path = validate_file_path(config_path)
        if not os.path.exists(valid_config_path):
            logging.error(f"ERROR: The indicated config file `{valid_config_path}` does NOT exist.")
            exit(1)

        self.config = VizMtmcConfig(**load_json_from_file(config_path))
        logging.info(f"Read config from {valid_config_path}")
        self.calibration_path = calibration_path

    def start_visualization(self) -> None:
        """
        Runs visualization of results from MTMC analytics

        :return: None
        ::

            visualization_app.start_visualization()
        """
        # Load frames from MTMC batch processing
        frames = None
        if self.config.setup.vizMode in {"frames", "behaviors", "mtmc_objects"}:
            frames = [Frame(**frame) for frame in
                    load_json_from_file_line_by_line(self.config.io.framesPath)]
            if (self.config.setup.vizMode == "mtmc_objects") and (self.config.setup.vizMtmcObjectsMode in {"grid", "topview"}):
                frames_id_1st = hash_frames(frames, self.config.io.selectedSensorIds)
            else:
                frames_sensor_1st = hash_frames(frames, self.config.io.selectedSensorIds, is_sensor_1st=True)

        # Load behaviors from MTMC batch processing
        behaviors = None
        if ((self.config.setup.vizMode == "frames") and self.config.plotting.vizFilteredFrames) or (self.config.setup.vizMode in {"behaviors", "mtmc_objects"}):
            behaviors = [Behavior(**behavior) for behavior in
                        load_json_from_file_line_by_line(self.config.io.behaviorsPath)]
            behaviors = hash_behaviors(behaviors, self.config.io.selectedSensorIds, self.config.io.selectedBehaviorIds)

        # Load MTMC objects from MTMC batch processing
        mtmc_objects = None
        if self.config.setup.vizMode == "mtmc_objects":
            mtmc_objects = [MTMCObject(**mtmc_object) for mtmc_object in
                            load_json_from_file_line_by_line(self.config.io.mtmcObjectsPath)]
            mtmc_objects = hash_mtmc_objects(mtmc_objects, self.config.io.selectedGlobalIds)

        # Load ground-truth bounding boxes
        ground_truth_bboxes = None
        if self.config.setup.vizMode == "ground_truth_bboxes":
            ground_truths = load_csv_from_file(self.config.io.groundTruthPath)
            ground_truth_bboxes = hash_ground_truth_bboxes(ground_truths, self.config.io.selectedSensorIds, self.config.io.videoDirPath)

        # Load ground-truth locations
        ground_truth_locations = None
        max_object_id = 0
        if self.config.setup.vizMode == "ground_truth_locations":
            ground_truths = load_csv_from_file(self.config.io.groundTruthPath)
            ground_truth_locations, max_object_id = hash_ground_truth_locations(ground_truths, self.config.io.selectedSensorIds, self.config.io.videoDirPath)

        # Visualize frames
        if self.config.setup.vizMode == "frames":
            output_video_dir = os.path.join(self.config.io.outputDirPath, f"viz_{self.config.setup.vizMode}")
            make_clean_dir(output_video_dir)
            viz_frames = VizFrames(self.config)
            if not self.config.setup.enableMultiprocessing:
                for sensor_id in frames_sensor_1st.keys():
                    viz_frames.plot(sensor_id, output_video_dir, frames_sensor_1st, behaviors)
            else:
                with Pool() as pool:
                    pool.starmap(viz_frames.plot,
                                zip(list(frames_sensor_1st), repeat(output_video_dir), repeat(frames_sensor_1st),
                                    repeat(behaviors)))
            logging.info("Visualization of frames complete\n")

        # Visualize behaviors
        elif self.config.setup.vizMode == "behaviors":
            output_video_dir = os.path.join(self.config.io.outputDirPath, f"viz_{self.config.setup.vizMode}")
            make_clean_dir(output_video_dir)
            viz_behaviors = VizBehaviors(self.config)
            if not self.config.setup.enableMultiprocessing:
                for sensor_id in behaviors.keys():
                    viz_behaviors.plot(sensor_id, output_video_dir, frames_sensor_1st, behaviors)
            else:
                with Pool() as pool:
                    pool.starmap(viz_behaviors.plot,
                                zip(list(behaviors), repeat(output_video_dir), repeat(frames_sensor_1st),
                                    repeat(behaviors)))
            logging.info("Visualization of behaviors complete\n")

        # Visualize MTMC objects
        elif self.config.setup.vizMode == "mtmc_objects":
            output_video_dir = os.path.join(self.config.io.outputDirPath, f"viz_{self.config.setup.vizMode}_in_{self.config.setup.vizMtmcObjectsMode}")
            make_clean_dir(output_video_dir)

            if self.config.setup.vizMtmcObjectsMode == "grid":
                viz_mtmc_objects_in_grid = VizMTMCObjectsInGrid(self.config)
                if not self.config.setup.enableMultiprocessing:
                    for global_id in mtmc_objects.keys():
                        viz_mtmc_objects_in_grid.plot(global_id, output_video_dir, frames_id_1st, behaviors, mtmc_objects)
                else:
                    with Pool() as pool:
                        pool.starmap(viz_mtmc_objects_in_grid.plot,
                                     zip(list(mtmc_objects), repeat(output_video_dir), repeat(frames_id_1st),
                                         repeat(behaviors), repeat(mtmc_objects)))
                logging.info("Visualization of MTMC objects in grid complete\n")

            elif self.config.setup.vizMtmcObjectsMode == "sequence":
                viz_mtmc_objects_in_sequence = VizMTMCObjectsInSequence(self.config)
                if not self.config.setup.enableMultiprocessing:
                    for global_id in mtmc_objects.keys():
                        viz_mtmc_objects_in_sequence.plot(global_id, output_video_dir, frames_sensor_1st, behaviors, mtmc_objects)
                else:
                    with Pool() as pool:
                        pool.starmap(viz_mtmc_objects_in_sequence.plot,
                                     zip(list(mtmc_objects), repeat(output_video_dir), repeat(frames_sensor_1st),
                                         repeat(behaviors), repeat(mtmc_objects)))
                logging.info("Visualization of MTMC objects in sequence complete\n")

            elif self.config.setup.vizMtmcObjectsMode == "topview":
                viz_mtmc_objects_in_topview = VizMTMCObjectsInTopview(self.config, self.calibration_path)
                if not self.config.setup.enableMultiprocessing:
                    for global_id in mtmc_objects.keys():
                        viz_mtmc_objects_in_topview.plot(global_id, output_video_dir, frames_id_1st, behaviors, mtmc_objects)
                else:
                    with Pool() as pool:
                        pool.starmap(viz_mtmc_objects_in_topview.plot,
                                    zip(list(mtmc_objects), repeat(output_video_dir), repeat(frames_id_1st),
                                        repeat(behaviors), repeat(mtmc_objects)))
                logging.info("Visualization of MTMC objects in top view complete\n")

            else:
                logging.error(f"ERROR: The mode {sanitize_string(self.config.setup.vizMtmcObjectsMode)} for the visualization of MTMC objects is unknown.")

        # Visualize ground-truth bounding boxes
        elif self.config.setup.vizMode == "ground_truth_bboxes":
            output_video_dir = os.path.join(self.config.io.outputDirPath, f"viz_{self.config.setup.vizMode}")
            make_clean_dir(output_video_dir)
            viz_ground_truth_bboxes = VizFrames(self.config)
            if not self.config.setup.enableMultiprocessing:
                for sensor_id in ground_truth_bboxes.keys():
                    viz_ground_truth_bboxes.plot(sensor_id, output_video_dir, ground_truth_bboxes, behaviors)
            else:
                with Pool() as pool:
                    pool.starmap(viz_ground_truth_bboxes.plot, zip(list(ground_truth_bboxes), repeat(output_video_dir), repeat(ground_truth_bboxes)))
            logging.info("Visualization of ground-truth bounding boxs complete\n")

        # Visualize ground-truth locations
        elif self.config.setup.vizMode == "ground_truth_locations":
            output_video_dir = os.path.join(self.config.io.outputDirPath, f"viz_{self.config.setup.vizMode}")
            make_clean_dir(output_video_dir)
            viz_ground_truth_locations = VizGroundTruthLocations(self.config, self.calibration_path)
            viz_ground_truth_locations.plot(output_video_dir, ground_truth_locations, max_object_id)
            logging.info("Visualization of ground-truth locations complete\n")

        else:
            logging.error(f"ERROR: The mode {sanitize_string(self.config.setup.vizMode)} for visualization is unknown.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=validate_file_path, default="resources/viz_mtmc_config.json",
                        action=ValidateFile, help="The input viz config file")
    parser.add_argument("--calibration", type=validate_file_path, default="resources/calibration_building_k.json",
                        action=ValidateFile, help="The input calibration file")
    args = parser.parse_args()
    visualization_app = VisualizationApp(args.config, args.calibration)
    visualization_app.start_visualization()
