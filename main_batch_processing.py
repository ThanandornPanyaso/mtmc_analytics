# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import argparse
import logging

from mdx.mtmc.schema import Frame, Behavior, MTMCObject
from mdx.mtmc.core.calibration import Calibrator
from mdx.mtmc.core.data import Loader, Preprocessor
from mdx.mtmc.core.clustering import Clusterer
from mdx.mtmc.core.evaluation import Evaluator
from mdx.mtmc.config import AppConfig
from mdx.mtmc.stream.state.people_height import StateManager as PeopleHeightEstimator
from mdx.mtmc.utils.io_utils import ValidateFile, validate_file_path, \
    load_json_from_file, load_json_from_file_line_by_line, write_pydantic_list_to_file, make_clean_dir

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)


class MTMCBatchProcessingApp:
    """
    Controller module for MTMC batch processing

    :param str config_path: path to the app config file
    :param str calibration_path: path to the calibration file in JSON format
    ::

        batch_processing_app = MTMCBatchProcessingApp(config_path, calibration_path)
    """

    def __init__(self, config_path: str, calibration_path: str) -> None:
        # Make sure the config file exists
        valid_config_path = validate_file_path(config_path)
        if not os.path.exists(valid_config_path):
            logging.error(f"ERROR: The indicated config file `{valid_config_path}` does NOT exist.")
            exit(1)

        self.config = AppConfig(**load_json_from_file(config_path))
        logging.info(f"Read config from {valid_config_path}\n")
        self.calibration_path = calibration_path
        self.calibrator = Calibrator(self.config)
        self.loader = Loader(self.config)
        self.preprocessor = Preprocessor(self.config)
        self.clusterer = Clusterer(self.config)
        self.evaluator = Evaluator(self.config)

    def start_batch_processing(self) -> None:
        """
        Runs MTMC batch processing and evaluation

        :return: None
        ::

            batch_processing_app.start_batch_processing()
        """
        if self.config.io.enableDebug:
            make_clean_dir(self.config.io.outputDirPath)
            frames_json_path = os.path.join(self.config.io.outputDirPath, "frames.json")
            behaviors_json_path = os.path.join(self.config.io.outputDirPath, "behaviors.json")
            mtmc_objects_json_path = os.path.join(self.config.io.outputDirPath, "mtmc_objects.json")

        # Calibrate sensors
        sensor_state_objects = self.calibrator.calibrate(self.calibration_path)
        self.preprocessor.set_sensor_state_objects(sensor_state_objects)
        logging.info("Calibration complete\n")

        # Load input data from the perception pipeline
        frames = None
        json_data_path = self.config.io.jsonDataPath
        protobuf_data_path = self.config.io.protobufDataPath
        if os.path.isfile(json_data_path):
            frames = self.loader.load_json_data_to_frames(json_data_path)
        elif os.path.isfile(protobuf_data_path):
            frames = self.loader.load_protobuf_data_to_frames(protobuf_data_path)
        else:
            logging.error(f"ERROR: The JSON data path {json_data_path} and "
                          f"protobuf data path {protobuf_data_path} do NOT exist.")
            exit(1)
        if self.config.io.enableDebug:
            write_pydantic_list_to_file(frames_json_path, frames)
            logging.info(f"Saved frame(s) to {validate_file_path(frames_json_path)}")
            frames = [Frame(**frame) for frame in
                      load_json_from_file_line_by_line(frames_json_path)]
        logging.info("Data loading complete\n")

        # Estimate people height
        self.people_height_estimator = PeopleHeightEstimator(self.config)
        self.people_height_estimator.set_sensor_state_objects(sensor_state_objects)
        self.people_height_estimator.estimate_people_height(frames)
        self.preprocessor.set_people_height_estimator(self.people_height_estimator)

        # Preprocess frames into behaviors and filter outliers
        behaviors = self.preprocessor.preprocess(frames)
        if not os.path.exists(self.config.io.groundTruthPath):
            del frames
        else:
            # Free unnecessary memory from frames
            for i in range(len(frames)):
                for j in range(len(frames[i].objects)):
                    frames[i].objects[j].embedding = None
        if self.config.io.enableDebug:
            write_pydantic_list_to_file(behaviors_json_path, behaviors)
            logging.info(f"Saved behavior(s) to {validate_file_path(behaviors_json_path)}")
            behaviors = [Behavior(**behavior) for behavior in
                         load_json_from_file_line_by_line(behaviors_json_path)]
        logging.info("Preprocessing complete\n")

        # Cluster behaviors into MTMC objects
        mtmc_objects, _, _ = self.clusterer.cluster(behaviors)
        del behaviors
        if self.config.io.enableDebug:
            write_pydantic_list_to_file(mtmc_objects_json_path, mtmc_objects)
            logging.info(f"Saved MTMC object(s) to {validate_file_path(mtmc_objects_json_path)}")
            mtmc_objects = [MTMCObject(**mtmc_object) for mtmc_object in
                            load_json_from_file_line_by_line(mtmc_objects_json_path)]
        logging.info("Clustering complete\n")

        # Evaluate MTMC results (if ground truth provided)
        if (len(mtmc_objects)) > 0 and os.path.exists(self.config.io.groundTruthPath):
            self.evaluator.evaluate(frames, mtmc_objects, sensor_state_objects)
            logging.info("Evaluation complete\n")
        del sensor_state_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=validate_file_path, default="resources/app_mtmc_config.json",
                        action=ValidateFile, help="The input app config file")
    parser.add_argument("--calibration", type=validate_file_path, default="resources/calibration_building_k.json",
                        action=ValidateFile, help="The input calibration file")
    args = parser.parse_args()
    batch_processing_app = MTMCBatchProcessingApp(args.config, args.calibration)
    batch_processing_app.start_batch_processing()
