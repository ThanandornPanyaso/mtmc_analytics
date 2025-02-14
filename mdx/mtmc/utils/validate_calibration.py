# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import argparse
import logging
import cv2
from typing import List, Dict, Tuple

from mdx.mtmc.schema import Sensor
from mdx.mtmc.core.calibration import Calibrator
from mdx.mtmc.config import VizMtmcConfig
from mdx.mtmc.utils.io_utils import ValidateFile, validate_file_path, \
    load_json_from_file, make_clean_dir

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)


class CalibrationValidator:
    """
    Module to validate calibration

    :param str config_path: path to the visualization config file
    :param str calibration_path: path to the calibration file in JSON format
    :param str output_dir_path: path to the output directory
    ::

        calibration_validator = CalibrationValidator(config_path, calibration_path, output_dir_path)
    """

    def __init__(self, config_path: str, calibration_path: str, output_dir_path: str):
        self.config = VizMtmcConfig(**load_json_from_file(config_path))
        calibrator = Calibrator(None)
        self.sensors = calibrator.load_calibration_file(calibration_path)
        self.sensors: Dict[str, Sensor] = {sensor.id: sensor for sensor in self.sensors}
        self.output_dir_path = output_dir_path
        make_clean_dir(output_dir_path)

    def plot_matched_coordinates(self) -> None:
        """
        Plots matched coordinates

        :return: None
        ::

            calibration_validator.plot_matched_coordinates()
        """
        # Read map image
        if not os.path.isfile(self.config.io.mapPath):
            logging.error("ERROR: The map image for top-view visualization does NOT exist.")
            exit(1)
        image_map = cv2.imread(self.config.io.mapPath)
        map_height, map_width = image_map.shape[:2]

        for video_name in sorted(os.listdir(self.config.io.videoDirPath)):
            image_map_copy = image_map.copy()
            sensor_id = video_name[:-4]

            # Read frame image
            input_video_path = os.path.join(self.config.io.videoDirPath, video_name)
            if not os.path.exists(input_video_path):
                logging.error(f"ERROR: The input video path `{validate_file_path(input_video_path)}` does NOT exist.")
                exit(1)
            video_capture = cv2.VideoCapture(input_video_path)
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            success, image_frame = video_capture.read()
            if not success:
                logging.error(f"ERROR: The input video `{validate_file_path(input_video_path)}` cannot be read.")
                exit(1)
            video_capture.release()

            # Extract image and global coordinates
            image_coordinates: List[Tuple[int]] = list()
            global_coordinates: List[Tuple[int]] = list()
            if sensor_id in self.sensors.keys():
                sensor = self.sensors[sensor_id]
                scale_factor = sensor.scaleFactor

                if (len(sensor.imageCoordinates) == 0) or (len(sensor.globalCoordinates) == 0):
                    logging.error(f"ERROR: The length of image coordinates or global coordinates is empty.")
                    exit(1)

                if len(sensor.imageCoordinates) != len(sensor.globalCoordinates):
                    logging.error(f"ERROR: The lengths of image coordinates and global coordinates do NOT match -- "
                                  f"{len(sensor.imageCoordinates)} != {len(sensor.globalCoordinates)}.")
                    exit(1)

                for coord in sensor.imageCoordinates:
                    image_coordinates.append((int(coord["x"]), int(coord["y"])))

                for coord in sensor.globalCoordinates:
                    global_coordinates.append((int(coord["x"] * scale_factor), map_height - int(coord["y"] * scale_factor) - 1))

            if (len(image_coordinates) == 0) or (len(global_coordinates) == 0):
                logging.error(f"ERROR: No matched sensor ID {sensor_id} is found.")
                exit(1)

            for i in range(len(image_coordinates)):
                # Plot indices
                cv2.putText(image_frame, str(i), image_coordinates[i], cv2.FONT_HERSHEY_DUPLEX,
                            2.0, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image_map_copy, str(i), global_coordinates[i], cv2.FONT_HERSHEY_DUPLEX,
                            2.0, (0, 0, 0), 2, cv2.LINE_AA)

                # Plot coordinates
                cv2.circle(image_frame, image_coordinates[i], 6, (255, 255, 255), -1)
                cv2.circle(image_map_copy, global_coordinates[i], 6, (255, 255, 255), -1)

            # Resize map image
            frame_scale = frame_width / float(map_width)
            image_map_copy = cv2.resize(image_map_copy, (frame_width, int(map_height * frame_scale)))

            # Concatenate images vertically
            image_output = cv2.vconcat([image_frame, image_map_copy])

            # Plot lines between matches
            for i in range(len(image_coordinates)):
                x = int(global_coordinates[i][0] * frame_scale)
                y = int((global_coordinates[i][1] * frame_scale) + frame_height)
                cv2.line(image_output, image_coordinates[i], (x, y), (0, 0, 0), 2)

            # Write output image
            cv2.imwrite(os.path.join(validate_file_path(self.output_dir_path), sensor_id + ".png"), image_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_config", type=validate_file_path, default="resources/viz_mtmc_config.json",
                        action=ValidateFile, help="The input viz config file")
    parser.add_argument("--calibration", type=validate_file_path, default="resources/calibration_building_k.json",
                        action=ValidateFile, help="The input calibration file")
    parser.add_argument("--output_dir", type=validate_file_path, default="results",
                        action=ValidateFile, help="The output directory")
    args = parser.parse_args()

    # Instantiate module
    calibration_validator = CalibrationValidator(args.viz_config, args.calibration, args.output_dir)

    # Plot matched coordinates
    calibration_validator.plot_matched_coordinates()
