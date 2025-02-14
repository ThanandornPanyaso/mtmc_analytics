# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import re
import argparse
import shutil
import json
import csv
import numpy as np
import pandas as pd
from typing import List, Set, Any


class ValidateFile(argparse.Action):
    """
    Module to validate files
    """
    def __call__(self, parser, namespace, values, option_string = None):
        
        if not os.path.exists(values):
            parser.error(f"Please enter a valid file path. Got: {values}")
        elif not os.access(values, os.R_OK):
            parser.error(f"File {values} doesn't have read access")
        setattr(namespace, self.dest, values)


def validate_file_path(input_string: str) -> str:
    """
    Validates whether the input string matches a file path pattern

    :param str input_string: input string
    :return: validated file path
    :rtype: str
    ::

        file_path = validate_file_path(input_string)
    """
    file_path_pattern = r"^[a-zA-Z0-9_\-\/.#]+$"

    if re.match(file_path_pattern, input_string):
        return input_string
    else:
        raise ValueError(f"Invalid file path: {input_string}")


def sanitize_string(input_string: str) -> str:
    """
    Sanitizes an input string

    :param str input_string: input string
    :return: sanitized string
    :rtype: str
    ::

        sanitized_string = sanitize_string(input_string)
    """
    # Allow alphanumeric characters, dots, slashes, underscores, hashes, and dashes
    return re.sub(r"[^a-zA-Z0-9\._/#-]", "_", input_string)


def load_json_from_file(file_path: str) -> Any:
    """
    Loads JSON data from a file

    :param str file_path: file path
    :return: data in the file
    :rtype: Any
    ::

        data = load_json_from_file(file_path)
    """
    valid_file_path = validate_file_path(file_path)
    with open(valid_file_path, "r") as f:
        data = json.load(f)
    return data


def load_json_from_file_line_by_line(file_path: str) -> List[Any]:
    """
    Loads JSON data from a file line by line

    :param str file_path: file path
    :return: list of data in the file
    :rtype: List[Any]
    ::

        data = load_json_from_file_line_by_line(file_path)
    """
    data: List[Any] = list()

    valid_file_path = validate_file_path(file_path)
    with open(valid_file_path, "r") as f:
        for line in f:
            line = line.rstrip('\n').replace("'", '"')
            data.append(load_json_from_str(line))

    return data


def load_json_from_str(string: str) -> Any:
    """
    Loads JSON data from a string

    :param str string: string
    :return: data in the string
    :rtype: Any
    ::

        data = load_json_from_str(string)
    """
    return json.loads(string)


def load_csv_from_file(file_path: str) -> List[List[str]]:
    """
    Loads from a CSV file

    :param str file_path: file path
    :return: data in the file
    :rtype: List[List[str]]
    ::

        data = load_csv_from_file(file_path)
    """
    valid_file_path = validate_file_path(file_path)
    with open(valid_file_path, "r") as file:
        csv_reader = csv.reader(file, delimiter=" ")
        data = [row for row in csv_reader]

    return data


def load_csv_to_dataframe_from_file(file_path: str, column_names: List[str]) -> pd.DataFrame:
    """
    Loads dataframe from a CSV file

    :param str file_path: file path
    :param List[str] column_names: column names
    :return: dataframe in the file
    :rtype: pd.DataFrame
    ::

        dataFrame = load_csv_to_dataframe_from_file(file_path, column_names)
    """
    data: List[List[str]] = list()

    valid_file_path = validate_file_path(file_path)
    with open(valid_file_path, "r") as file:
        for line in file:
            row = re.split(r"\s+|\t+|,", line.strip())
            data.append(row)

    data_array = np.array(data, dtype=np.float32)
    data_dict = {column_names[i]: data_array[:, i] for i in range(len(column_names))}

    return pd.DataFrame(data_dict)


def write_dataframe_to_csv_file(file_path: str, data: pd.DataFrame, delimiter: str = " ") -> None:
    """
    Writes dataframe to a CSV file

    :param str file_path: file path
    :param pd.DataFrame data: dataframe to be written
    :param str delimiter: delimiter of the CSV file
    :return: None
    ::

        write_dataframe_to_csv_file(file_path, data, delimiter)
    """
    data.to_csv(file_path, sep=delimiter, index=False, header=False)


def write_pydantic_list_to_file(file_path: str, data: Any) -> None:
    """
    Writes a list of pydantic data to a file

    :param str file_path: file path
    :param Any data: data to be written
    :return: None
    ::

        write_pydantic_list_to_file(file_path, data)
    """
    with open(file_path, "w") as f:
        for datum in data:
            f.write(f"{datum.json()}\n")


def make_clean_dir(dir_path: str) -> None:
    """
    Makes a clean directory

    :param str dir_path: directory path
    :return: None
    ::

        make_clean_dir(dir_path)
    """
    valid_dir_path = validate_file_path(dir_path)
    if os.path.exists(valid_dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    if not os.path.isdir(valid_dir_path):
        os.makedirs(validate_file_path(dir_path))


def make_dir(dir_path: str) -> None:
    """
    Makes a directory without removing other files

    :param str dir_path: directory path
    :return: None
    ::

        make_dir(dir_path)
    """
    valid_dir_path = validate_file_path(dir_path)
    if not os.path.isdir(valid_dir_path):
        os.makedirs(validate_file_path(dir_path))


def make_seq_maps_file(file_dir: str, sensor_ids: Set[str], benchmark: str, split_to_eval: str) -> None:
    """
    Makes a sequence-maps file used by TrackEval library

    :param str file_dir: file path
    :param Set(str) sensor_ids: names of sensors
    :param str benchmark: name of the benchmark
    :param str split_to_eval: name of the split of data
    :return: None
    ::

        make_seq_maps_file(file_dir, sensor_ids)
    """
    make_clean_dir(file_dir)
    file_name = benchmark + "-" +split_to_eval + ".txt"
    seq_maps_file = file_dir +  "/" + file_name
    f = open(seq_maps_file, "w")
    f.write("name\n")

    for name in sensor_ids:
        sensor_name = name + "\n"
        f.write(sensor_name)
    f.write("MTMC")
    f.close()


def make_seq_ini_file(gt_dir: str, camera: str, seq_length: int) -> None:
    """
    Makes a sequence-ini file used by TrackEval library

    :param str gt_dir: file path
    :param str camera: Name of a single sensor 
    :param int seq_length: Number of frames

    :return: None
    ::

        make_seq_ini_file(gt_dir, camera, seq_length)
    """
    ini_file_name = gt_dir + "/seqinfo.ini"
    f = open(ini_file_name, "w")
    f.write("[Sequence]\n")
    name= "name=" +str(camera)+ "\n"
    f.write(name)
    f.write("imDir=img1\n")
    f.write("frameRate=30\n")
    seq = "seqLength=" + str(seq_length) + "\n"
    f.write(seq)
    f.write("imWidth=1920\n")
    f.write("imHeight=1080\n")
    f.write("imExt=.jpg\n")
    f.close()


def remove_dir(dir_path: str) -> None:
    """
    Removes a directory

    :param str dir_path: directory path
    :return: None
    ::

        remove_dir(dir_path)
    """
    valid_dir_path = validate_file_path(dir_path)
    if os.path.exists(valid_dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)


def move_files(source_dir: str, destination_dir: str) -> None:
    """
    Moves contents of a folder from source_dir to destination_dir (non-recursive)

    :param str source_dir: source directory path
    :param str destination_dir: destination directory path
    :return: None
    ::

        move_files(source_dir, destination_dir)
    """
    valid_source_dir_path = validate_file_path(source_dir)
    valid_destination_dir_path = validate_file_path(destination_dir)
    if os.path.exists(valid_source_dir_path) and os.path.exists(valid_destination_dir_path):
        for filename in os.listdir(valid_source_dir_path):
            shutil.move(os.path.join(valid_source_dir_path, filename),
                        os.path.join(valid_destination_dir_path, filename))
