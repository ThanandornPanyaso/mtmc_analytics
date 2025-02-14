# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import sys
import os
import time
import pytz
import json
import logging
import argparse
from datetime import datetime
from utils import *

def synchronize_raw_data_files(input_folder: str, output_folder: str, fps: float, metadata_format: str):
    """
    This function reads the files from the 'input_folder' variable and outputs a synchronized file
    named 'synchronized_output.json'. Individual files will be stored in 'output_folder'.

    :param str input_folder: The name of the directory which contains the raw data logs.
    :param str output_folder: The directory containing individual logs of each sensor.
    :param float fps: Frame per seconds in float to determine the timestamp interval.
    :param str metadata_format: Type of the metadata format (protobuf or json).

    """
    if metadata_format == "json":
        timestamp_field = "@timestamp"
        file_extension = "json"
    elif metadata_format == "protobuf":
        timestamp_field = "timestamp"
        file_extension = "txt"
    else:
        logging.info(f"Incorrect metadata format type detected. Exiting")
        exit()

    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
 
    timezone = pytz.timezone("America/Los_Angeles")
    base_timestamp = time.time()

    output_file_name = os.path.join(output_folder, f"synchronized_output.{file_extension}")
    output_file = open(output_file_name, "w")

    messages = {}

    logging.info(f"Collecting all raw data logs...")
    for raw_messages_name in os.listdir(input_folder):
        raw_messages_path = os.path.join(input_folder, raw_messages_name)
        if os.path.isdir(raw_messages_path):
            continue
        output_messages_path = os.path.join(output_folder, raw_messages_name)
        output_messages_file = open(output_messages_path, "w")
        with open(raw_messages_path, "r") as raw_messages_file:
            for message_line in raw_messages_file:
                message_line = message_line.rstrip("\n")
                message_line = message_line.replace("'", '"')
                try:
                    message_idx = message_line.index("{")
                except ValueError:
                    logging.error("The message does NOT exist.")
                    message_idx = -1
                if message_idx == -1:
                    continue
                message = json.loads(message_line[message_idx:])
                frame_num = int(message["id"])
                curr_timestamp = base_timestamp + (frame_num / fps)
                message[timestamp_field] = datetime.fromtimestamp(curr_timestamp, timezone).isoformat("T", "milliseconds")[:-6] + "Z"
                messages[message[timestamp_field] + " - " + raw_messages_name] = message
                output_messages_file.write(json.dumps(message) + "\n")
        output_messages_file.close()

    for message_key in sorted(list(messages)):
        output_file.write(json.dumps(messages[message_key]) + "\n")

    output_file.close()
    logging.info(f"Finished synchronization process.")
    logging.info(f"Synchronized output file can be found here: {output_file_name}")

if __name__ == "__main__":

    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True) 
    parser.add_argument('--output_folder', required=True) 
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--metadata_format', default="json", choices=["json", "protobuf"])

    # Parse all arguments
    args = parser.parse_args()

    # Synchronize the logs 
    synchronize_raw_data_files(args.input_folder, args.output_folder, args.fps, args.metadata_format)