# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import json
import logging
import argparse
from typing import Dict


def convert_single_log_to_multi_cam_logs(input_file: str, output_dir: str) -> None:
    """
    This function converts a single log file containing metadata for multiple sensors to multiple log files,
    each corresponding to a sensor, based on the sensor ID.

    :param str input_file: The path to the input log file containing metadata for multiple sensors.
    :param str output_dir: The output directory where separate log files for each sensor will be saved.

    This function reads the input log file and extracts the metadata for each sensor. 
    It then maps each sensor to its corresponding log file based on the sensor ID. 
    The function writes the metadata lines for each sensor to its respective log file.
    """
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    input_log_file = open(input_file, 'r')

    sensor_id_2_file_pointer: Dict[str, _io.TextIOWrapper] = dict()

    logging.info(f"Reading the unsynchronized input file: {input_file}")

    for line in input_log_file:
        metadata_line = line.rstrip("\n")
        metadata = json.loads(metadata_line)
        sensor_id = metadata["sensorId"]

        if sensor_id not in sensor_id_2_file_pointer:
            sensor_file_name = sensor_id + ".txt"
            sensor_abs_file_path = os.path.join(output_dir, sensor_file_name)
            sensor_id_2_file_pointer[sensor_id] = open(sensor_abs_file_path, "a")
        sensor_id_2_file_pointer[str(sensor_id)].write(line)

    input_log_file.close()

    for sensor_id, file_pointer in sensor_id_2_file_pointer.items():
        file_pointer.close()

    logging.info(f"Generated individual raw data file for each sensor id.")
    logging.info(f"Output files can be found here: {output_dir}")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True) 
    parser.add_argument('-o', '--output_dir', required=True) 

    # Parse all arguments
    args = parser.parse_args()

    # Get multi-cam logs
    convert_single_log_to_multi_cam_logs(args.input_file, args.output_dir)