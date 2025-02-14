# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import sys
import os
import time
import json
import logging
import argparse
from utils import *

def remove_empty_logs(input_file_name, output_file_name):
    """
    This function reads the file 'input_file_name' and removes logs with empty sensorIDs.
    It then outputs a synchronized file as 'output_file_name'.

    :param str input_file_name: The name of the directory which contains the data log.
    :param str output_file_name: The name of the directory where the output log will be stored.

    """
    output_file = open(output_file_name, "w")
    
    with open(input_file_name) as f:
        for line in f:
            line = line.rstrip() 
            frame_log = json.loads(line)
            sensor_id = frame_log["sensorId"]

            if sensor_id != "":
                frame_log = json.dumps(frame_log)+'\n'
                output_file.write(frame_log)

    output_file.close()

if __name__ == "__main__":

    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True) 
    parser.add_argument('--output_file', required=True) 

    # Parse all arguments
    args = parser.parse_args()

    # Remove logs with empty sensor IDs
    logging.info(f"Removing empty logs...")
    remove_empty_logs(args.input_file, args.output_file)
    logging.info(f"Final synchronized output file can be found here: {args.output_file}")
