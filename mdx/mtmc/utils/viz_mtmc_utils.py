# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import glob
import logging
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Deque, Tuple, Optional
from collections import deque
from datetime import datetime

from mdx.mtmc.schema import Bbox, Object, Frame, Behavior, MTMCObject


def get_predefined_bgr_color(idx: int) -> Tuple[int, int, int]:
    """
    Gets pre-defined BGR color

    :param int idx: index
    :return: BGR color
    :rtype: Tuple[int, int, int]
    ::

        b, g, r = get_predefined_bgr_color(idx)
    """
    if idx % 4 == 0:
        return 239, 120, 223

    if idx % 4 == 1:
        return 0, 185, 118

    if idx % 4 == 2:
        return 50, 50, 240

    if idx % 4 == 3:
        return 213, 145, 113


def get_random_bgr_color(idx: int, color_gap_min: int = 64) -> Tuple[int, int, int]:
    """
    Gets random BGR color

    :param int idx: index
    :param int color_gap_min: minimum color gap
    :return: BGR color
    :rtype: Tuple[int, int, int]
    ::

        b, g, r = get_random_bgr_color(idx, color_gap_min)
    """
    if idx > 20:
        b, g, r = (255, 255, 255)
        while 255 - ((b + g + r) / 3) < color_gap_min:
            b, g, r = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    else:
        b = 0 if idx % 6 in {0, 1, 3} else min(255, (256 >> (idx // 6)))
        b += random.randint(-5, 5)
        b = max(0, min(255, b))
        g = 0 if idx % 6 in {0, 2, 4} else min(255, (256 >> (idx // 6)))
        g += random.randint(-5, 5)
        g = max(0, min(255, g))
        r = 0 if idx % 6 in {1, 2, 5} else min(255, (256 >> (idx // 6)))
        r += random.randint(-5, 5)
        r = max(0, min(255, r))

    return b, g, r


def smooth_tail(tail: Optional[Deque], tail_window: int) -> Deque:
    """
    Smooths tail by moving averages in both x-axis and y-axis

    :param Optional[Deque] tail: tail of the object
    :param int tail_window: window for smoothing the tail
    :return: moving averages of the tail
    :rtype: Deque
    ::

        smoothed_tail = smooth_tail(tail, tail_window)
    """
    if not tail:
        return tail

    if isinstance(tail, deque):
        tail = np.asarray(list(tail))
    else:
        tail = np.asarray(tail)

    tail_window = min(tail_window, len(tail))
    moving_averages_x: List[int] = list()
    moving_averages_y: List[int] = list()
    i = 0
    while i < len(tail) - tail_window + 1:
        window_average_x = int(np.sum(tail[i:i+tail_window, 0]) / tail_window)
        moving_averages_x.append(window_average_x)
        window_average_y = int(np.sum(tail[i:i+tail_window, 1]) / tail_window)
        moving_averages_y.append(window_average_y)
        i += 1
    moving_averages = list(zip(moving_averages_x, moving_averages_y))

    return deque(moving_averages)


def hash_frames(frames: List[Frame], selected_sensor_ids: List[str], is_sensor_1st: bool = False) -> \
    Dict[str, Dict[str, Dict[str, Object]]]:
    """
    Hashes frames to a dict

    :param List[Frame] frames: list of frames
    :param List[str] selected_sensor_ids: selected sensor IDs
    :param bool is_sensor_1st: is sensor ID the first level
    :return: hashed dictionary of frames
    :rtype: Dict[str,Dict[str,Dict[str,Object]]]
    ::

        frame_dict = hash_frames(frames, selected_sensor_ids, is_sensor_1st)
    """
    frame_dict: Dict[str, Dict[str, Dict[str, Object]]] = dict()
    for frame in frames:
        frame_id = frame.id
        sensor_id = frame.sensorId

        if (len(selected_sensor_ids) > 0) and (sensor_id not in selected_sensor_ids):
                continue

        # frame IDs are at the first level
        if not is_sensor_1st:
            if frame_id not in frame_dict.keys():
                frame_dict[frame_id] = dict()

            frame_dict[frame_id][sensor_id]= dict()

            for object_instance in frame.objects:
                frame_dict[frame_id][sensor_id][object_instance.id] = object_instance

        # sensor IDs are at the first level
        else:
            if sensor_id not in frame_dict.keys():
                frame_dict[sensor_id] = dict()

            frame_dict[sensor_id][frame_id]= dict()

            for object_instance in frame.objects:
                frame_dict[sensor_id][frame_id][object_instance.id] = object_instance

    return frame_dict


def hash_behaviors(behaviors: List[Behavior], selected_sensor_ids: List[str], selected_behavior_ids: List[str] = list()) -> \
    Dict[str, Dict[str, Behavior]]:
    """
    Hashes behaviors to a dict

    :param List[Behavior] behaviors: list of behaviors
    :param List[str] selected_sensor_ids: selected sensor IDs
    :param List[str] selected_behavior_ids: selected behavior IDs
    :return: hashed dictionary of behaviors
    :rtype: Dict[str,Dict[str,Behavior]]
    ::

        behavior_dict = hash_behaviors(behaviors, selected_sensor_ids, selected_behavior_ids)
    """
    behavior_dict: Dict[str, Dict[str, Behavior]] = dict()
    for behavior in behaviors:
        behavior_id = behavior.id
        sensor_id = behavior.sensorId

        if (len(selected_behavior_ids) > 0) and (behavior_id not in selected_behavior_ids):
                continue

        if (len(selected_sensor_ids) > 0) and (sensor_id not in selected_sensor_ids):
                continue

        if sensor_id not in behavior_dict:
            behavior_dict[sensor_id] = dict()

        behavior_dict[sensor_id][behavior.key] = behavior

    return behavior_dict


def hash_mtmc_objects(mtmc_objects: List[MTMCObject], selected_global_ids: List[str]) -> \
    Dict[str, Dict[str, Behavior]]:
    """
    Hashes MTMC objects to a dict

    :param List[MTMCObject] mtmc_objects: list of MTMC objects
    :param List[str] selected_global_ids: selected global IDs
    :return: hashed dictionary of matched behaviors
    :rtype: Dict[str,Dict[str,Behavior]]
    ::

        mtmc_object_dict = hash_mtmc_objects(mtmc_objects, selected_global_ids)
    """
    mtmc_object_dict: Dict[str, Dict[str, Behavior]] = dict()
    for mtmc_object in mtmc_objects:
        global_id = mtmc_object.globalId

        if (len(selected_global_ids) > 0) and (global_id not in selected_global_ids):
                continue

        if global_id not in mtmc_object_dict.keys():
            mtmc_object_dict[global_id] = dict()

        for matched_behavior in mtmc_object.matched:
            mtmc_object_dict[global_id][matched_behavior.key] = matched_behavior

    return mtmc_object_dict


def hash_ground_truth_bboxes(ground_truths: List[List[str]], selected_sensor_ids: List[str], video_dir_path: str) -> Dict[str, Dict[str, Dict[str, Object]]]:
    """
    Hashes ground-truth bounding boxes to a dict

    :param List[List[str]] ground_truths: list of ground truths
    :param List[str] selected_sensor_ids: selected sensor IDs
    :param str video_dir_path: path to the directory of videos
    :return: hashed dictionary of ground-truth bounding boxes
    :rtype: Dict[str,Dict[str,Dict[str,Object]]]
    ::

        ground_truth_dict = hash_ground_truth_bboxes(ground_truths, selected_sensor_ids, video_dir_path)
    """
    sensor_ids = sorted([os.path.splitext(os.path.basename(video_name))[0] for video_name in glob.glob(os.path.join(video_dir_path, '*.mp4'))])
    ground_truth_dict: Dict[str, Dict[str, Dict[str, Object]]] = dict()
    for ground_truth in ground_truths:
        sensor_idx = int(ground_truth[0]) - 1
        if sensor_idx < 0:
            logging.error(f"ERROR: The sensor index {sensor_idx} is smaller than 0.")
            exit(1)
        if sensor_idx >= len(sensor_ids):
            logging.error(f"ERROR: The sensor index {sensor_idx} is larger than the number of sensor IDs {len(sensor_ids)}.")
            exit(1)
        sensor_id = sensor_ids[sensor_idx]
        object_id = ground_truth[1]
        frame_id = ground_truth[2]

        if (len(selected_sensor_ids) > 0) and (sensor_id not in selected_sensor_ids):
                continue

        if sensor_id not in ground_truth_dict.keys():
            ground_truth_dict[sensor_id] = dict()

        if frame_id not in ground_truth_dict[sensor_id].keys():
            ground_truth_dict[sensor_id][frame_id] = dict()

        bbox_xmin = float(ground_truth[3])
        bbox_ymin = float(ground_truth[4])
        bbox_width = float(ground_truth[5])
        bbox_height = float(ground_truth[6])
        bbox = Bbox(leftX=bbox_xmin, topY=bbox_ymin, rightX=bbox_xmin+bbox_width-1.0, bottomY=bbox_ymin+bbox_height-1.0)

        ground_truth_dict[sensor_id][frame_id][object_id] = Object(id=object_id, bbox=bbox,
                                                                   type="Person", confidence=1.0,
                                                                   info=None, embedding=None)

    return ground_truth_dict


def hash_ground_truth_locations(ground_truths: List[List[str]], selected_sensor_ids: List[str], video_dir_path: str) -> Tuple[Dict[str, Dict[str, List[float]]], int]:
    """
    Hashes ground-truth locations to a dict

    :param List[List[str]] ground_truths: list of ground truths
    :param List[str] selected_sensor_ids: selected sensor IDs
    :param str video_dir_path: path to the directory of videos
    :return: hashed dictionary of ground-truth locations and max object ID
    :rtype: Tuple[Dict[str,Dict[str,List[float]]],int]
    ::

        ground_truth_dict, max_object_id = hash_ground_truth_locations(ground_truths, selected_sensor_ids, video_dir_path)
    """
    max_object_id = -1
    sensor_ids = sorted([os.path.splitext(os.path.basename(video_name))[0] for video_name in glob.glob(os.path.join(video_dir_path, '*.mp4'))])
    ground_truth_dict: Dict[str, Dict[str, List[float]]] = dict()
    for ground_truth in ground_truths:
        sensor_idx = int(ground_truth[0]) - 1
        if sensor_idx < 0:
            logging.error(f"ERROR: The sensor index {sensor_idx} is smaller than 0.")
            exit(1)
        if sensor_idx >= len(sensor_ids):
            logging.error(f"ERROR: The sensor index {sensor_idx} is larger than the number of sensor IDs {len(sensor_ids)}.")
            exit(1)
        sensor_id = sensor_ids[sensor_idx]
        object_id = ground_truth[1]
        frame_id = ground_truth[2]

        if (len(selected_sensor_ids) > 0) and (sensor_id not in selected_sensor_ids):
                continue

        if frame_id not in ground_truth_dict.keys():
            ground_truth_dict[frame_id]= dict()

        location = [float(ground_truth[7]), float(ground_truth[8])]
        if object_id not in ground_truth_dict[frame_id].keys():
            ground_truth_dict[frame_id][object_id] = location
        elif (location[0] != ground_truth_dict[frame_id][object_id][0]) or (location[1] != ground_truth_dict[frame_id][object_id][1]):
            logging.error(f"ERROR: The location for frame ID {frame_id} and object ID {object_id} is not consistent across sensors: {location} != {ground_truth_dict[frame_id][object_id]}.")
            exit(1)

        if int(object_id) > max_object_id:
            max_object_id = int(object_id)

    return ground_truth_dict, max_object_id


def get_frame_objects_and_timestamps(frames: List[Frame]) -> Tuple[Dict[Tuple[str, int, int], Object], Dict[Tuple[str, int], datetime]]:
    """
    Gets objects and timestamps from frames

    :param List[Frame] frames: list of frames
    :return: dictionaries of objects and timestamps
    :rtype: Tuple[Dict[Tuple[str,int,int],Object],Dict[Tuple[str,int],datetime]]
    ::

        objects, timestamps = get_frame_objects_and_timestamps(frames)
    """
    objects: Dict[Tuple[str, int, int], Object] = dict()
    timestamps: Dict[Tuple[str, int], datetime] = dict()
    for frame in frames:
        sensor_id = frame.sensorId
        frame_id = int(frame.id)
        timestamps[(sensor_id, frame_id)] = frame.timestamp
        for object_instance in frame.objects:
            objects[(sensor_id, frame_id, int(object_instance.id))] = object_instance

    return objects, timestamps


def map_behaviors_from_keys(behaviors: List[Behavior]) -> Dict[str, Behavior]:
    """
    Maps behavior keys to behaviors

    :param List[Behavior] behaviors: list of behaviors
    :return: map from behavior keys to behaviors
    :rtype: Dict[str,Behavior]
    ::

        behavior_dict = map_behaviors_from_keys(behaviors)
    """
    behavior_dict: Dict[str, Behavior] = dict()
    for behavior in behaviors:
        behavior_dict[behavior.key] = behavior

    return behavior_dict


def plot_object(image_frame: np.array, bbox: Bbox, convex_hull: Optional[List[int]], foot_pixel: List[str], bbox_label: str, bbox_color: Tuple[int, int, int]) -> np.array:
    """
    Plots an object on a frame image

    :param np.array image_frame: image frame
    :param Bbox bbox: bounding box
    :param Optional[List[int]] convex_hull: convex hull
    :param List[int] foot_pixel: foot location in pixel
    :param str bbox_label: bbox label
    :param Tuple[int,int,int] bbox_color: bbox color
    :return: plotted image frame
    :rtype: np.array
    ::

        image_frame = plot_object(image_frame, bbox, convex_hull, foot_pixel, bbox_label, bbox_color)
    """
    text_face = cv2.FONT_HERSHEY_DUPLEX
    text_scale = 2.0
    text_thickness = 2

    bbox_left_x = int(bbox.leftX)
    bbox_top_y = int(bbox.topY)
    bbox_right_x = int(bbox.rightX)
    bbox_bottom_y = int(bbox.bottomY)
    bbox_label_size = cv2.getTextSize(bbox_label, text_face, text_scale, text_thickness)[0]

    # Plot bounding box or convex hull
    if convex_hull is None:
        cv2.rectangle(image_frame, (bbox_left_x, bbox_top_y), (bbox_right_x, bbox_bottom_y), bbox_color, 3)
    else:
        bbox_center = [(bbox_left_x + bbox_right_x) // 2, (bbox_top_y + bbox_bottom_y) // 2]
        convex_hull_pixels = []
        for i in range(0, len(convex_hull), 2):
            if (i + 1) < len(convex_hull):
                convex_hull_pixels.append([int(convex_hull[i]) + bbox_center[0], int(convex_hull[i+1]) + bbox_center[1]])
        del convex_hull
        cv2.polylines(image_frame, [np.array(convex_hull_pixels)], True, bbox_color, 3)
        del convex_hull_pixels

    # Plot label
    if bbox_top_y - bbox_label_size[1] - 8 >= 0:
        cv2.rectangle(image_frame, (bbox_left_x - 2, bbox_top_y - bbox_label_size[1] - 8), (bbox_left_x + bbox_label_size[0] + 3, bbox_top_y), bbox_color, -1)
        cv2.putText(image_frame, bbox_label, (bbox_left_x, bbox_top_y - 3), text_face, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(image_frame, (bbox_left_x, bbox_top_y - 2), (bbox_left_x + bbox_label_size[0] + 3, bbox_top_y + bbox_label_size[1] + 10), bbox_color, -1)
        cv2.putText(image_frame, bbox_label, (bbox_left_x, bbox_top_y + bbox_label_size[1] + 5), text_face, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    # Plot foot point
    cv2.circle(image_frame, foot_pixel, 9, (0, 0, 255), -1)

    return image_frame


def plot_frame_label(image_frame: np.array, frame_label: str) -> np.array:
    """
    Plots frame label on a frame image

    :param np.array image_frame: image frame
    :param str frame_label: frame label
    :return: plotted image frame
    :rtype: np.array
    ::

        image_frame = plot_frame_label(image_frame, frame_label)
    """
    text_face = cv2.FONT_HERSHEY_DUPLEX
    text_scale = 2.0
    text_thickness = 2

    frame_labels = frame_label.split("\n")
    num_frame_labels = len(frame_labels)
    frame_label_width_max = 0
    frame_label_height_max = 0

    for frame_label in frame_labels:
        frame_label_size = cv2.getTextSize(frame_label, text_face, text_scale, text_thickness)[0]
        if frame_label_size[0] > frame_label_width_max:
            frame_label_width_max = frame_label_size[0]
        if frame_label_size[1] > frame_label_height_max:
            frame_label_height_max = frame_label_size[1]

    cv2.rectangle(image_frame, (0, 0), (frame_label_width_max + 24, (frame_label_height_max  + 44) * num_frame_labels), (66, 66, 66), -1)

    for i in range(num_frame_labels):
        cv2.putText(image_frame, frame_labels[i], (14, (frame_label_height_max * (i + 1)) + (16 * i) + 20), 
                    text_face, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return image_frame


def plot_histograms(x_axis: List[float], arr_neg: np.array, arr_pos: np.array,
                    interval: float = 0.02, title: str = "", output_dir_path: str = "",
                    mode: Optional[str] = None, labels: Optional[List[str]] = None) -> None:
    """
    Plots histograms

    :param List[float] x_axis: x-axis
    :param np.array arr_neg: negative array
    :param np.array arr_pos: positive array
    :param float interval: interval of the x-axis
    :param str title: title of the plot
    :param str output_dir_path: path to the output directory
    :param Optional[str] mode: histogram mode (mean, median, or individual)
    :param Optional[List[str]] labels: custom labels
    :return: None
    ::

        plot_histograms(x_axis, arr_neg, arr_pos, interval, title, output_dir_path, mode, labels)
    """
    plt.figure(figsize=(12, 8))

    if mode == "individual":
        label = "Negative bboxes"
    elif mode == "custom":
        label = labels[0]
    else:
        label = "Negative behaviors"
    plt.bar(x_axis, arr_neg, label=label,
            width=interval, align="edge", alpha=0.5, ls="dotted", lw=0.1,
            edgecolor="black")

    if mode == "individual":
        label = "Positive bboxes"
    elif mode == "custom":
        label = labels[1]
    else:
        label = "Positive behaviors"
    plt.bar(x_axis, arr_pos, label=label,
            width=interval, align="edge", alpha=0.5, ls="dashed", lw=0.1,
            edgecolor="black")

    plt.legend()
    plt.grid()
    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.title(f"Histograms of {title}")
    plt.savefig(os.path.join(output_dir_path, f"histograms_{title}.png"))
    # plt.show()


def plot_roc(tpr: List[float], fpr: List[float],
             tpr_optim: float, fpr_optim: float, 
             title: str, output_dir_path: str) -> None:
    """
    Plots ROC curve

    :param List[float] tpr: TPR values
    :param List[float] fpr: FPR values
    :param float tpr_optim: optimum TPR value
    :param float fpr_optim: optimum FPR value
    :param str title: title of the plot
    :param str output_dir_path: path to the output directory
    :return: None
    ::

        plot_roc(tpr, fpr, tpr_optim, fpr_optim, title, output_dir_path)
    """
    zipped_coordinates = sorted(list(zip(fpr, tpr)))
    x_coord = [zc[0] for zc in zipped_coordinates]
    y_coord = [zc[1] for zc in zipped_coordinates]

    plt.figure(figsize=(12, 8))
    plt.plot(x_coord, y_coord, linewidth=2)
    plt.scatter([fpr_optim], [tpr_optim], s=150, c="r", marker="*")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC curve of {title}")
    plt.savefig(os.path.join(output_dir_path, f"roc_{title}.png"))
    # plt.show()
