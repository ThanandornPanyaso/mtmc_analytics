# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import shutil
import logging
import cv2
import ffmpeg
import numpy as np
from typing import List, Dict, Set, Optional
from collections import deque

from mdx.mtmc.config import VizMtmcConfig
from mdx.mtmc.schema import Bbox, Object, Behavior
from mdx.mtmc.utils.io_utils import validate_file_path, sanitize_string
from mdx.mtmc.utils.viz_mtmc_utils import get_predefined_bgr_color, smooth_tail, \
    plot_object, plot_frame_label

DIMMING_FACTOR = 0.5


class VizMTMCObjectsInGrid:
    """
    Module to visualize MTMC objects in grid

    :param VizMtmcConfig config: configuration for visualization
    ::

        visualizer = VizMTMCObjectsInGrid(config)
    """

    def __init__(self, config: VizMtmcConfig) -> None:
        self.config: VizMtmcConfig = config

        # num_videos = len(os.listdir(self.config.io.videoDirPath))
        # if num_videos < self.config.plotting.gridLayout[0] * self.config.plotting.gridLayout[1]:
        #     logging.error(f"ERROR: The number of videos {num_videos} is less than the grid layout "
        #                   f"({sanitize_string(str(self.config.plotting.gridLayout[0]))}, {sanitize_string(str(self.config.plotting.gridLayout[1]))}).")
        #     exit(1)

    def plot(self, global_id: str,
             output_video_dir: str,
             frames: Dict[str, Dict[str, Dict[str, Object]]],
             behaviors: Dict[str, Dict[str, Behavior]],
             mtmc_objects: Dict[str, Dict[str, Behavior]]) -> None:
        """
        Visualizes MTMC objects in grid

        :param str global_id: global ID of an MTMC object
        :param str output_video_dir: output directory for videos
        :param Dict[str,Dict[str,Dict[str,Object]]] frames: frames extracted from raw data
        :param Dict[str,Dict[str,Behavior]] behaviors: behaviors, i.e., tracklets
        :param Dict[str,Dict[str,Behavior]] mtmc_objects: MTMC objects, i.e., matches
        :return: None
        ::

            visualizer.plot(global_id, output_video_dir, frames, behaviors, mtmc_objects)
        """
        # Get the index of the global ID
        global_ids = sorted([int(global_id) for global_id in mtmc_objects.keys()])
        global_idx = global_ids.index(int(global_id))

        # Set sensor IDs
        num_sensors = self.config.plotting.gridLayout[0] * self.config.plotting.gridLayout[1]
        if len(list(behaviors)) != num_sensors:
            logging.warning(f"WARNING: The number of sensors {list(behaviors)} does not match the grid layout {self.config.plotting.gridLayout} -- "
                            f"By default only the top {num_sensors} sensors will be plotted.")
            sensor_ids: Set[str] = set()
            for frame_id in frames.keys():
                for sensor_id in frames[str(frame_id)].keys():
                    sensor_ids.add(sensor_id)
            sensor_ids = sorted(list(sensor_ids))[:num_sensors]
        else:
            sensor_ids = sorted(list(behaviors))

        # Set global starting and ending frames
        first_behavior_key = list(mtmc_objects[global_id])[0]
        global_start_frame = int(mtmc_objects[global_id][first_behavior_key].startFrame)
        global_end_frame = int(mtmc_objects[global_id][first_behavior_key].endFrame)
        for behavior_key in mtmc_objects[global_id].keys():
            matched_behavior = mtmc_objects[global_id][behavior_key].copy()
            if int(matched_behavior.startFrame) < global_start_frame:
                global_start_frame = int(matched_behavior.startFrame)
            if int(matched_behavior.endFrame) > global_end_frame:
                global_end_frame = int(matched_behavior.endFrame)

        # Initialize video writer
        input_video_path = os.path.join(self.config.io.videoDirPath, f"{list(behaviors)[0]}.mp4")
        if not os.path.exists(input_video_path):
            logging.error(f"ERROR: The input video path `{input_video_path}` does NOT exist.")
            exit(1)
        video_capture = cv2.VideoCapture(input_video_path)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        if self.config.setup.ffmpegRequired:
            output_video_name = f"object_id_{global_id}_mpeg4.mp4"
        else:
            output_video_name = f"object_id_{global_id}.mp4"
        grid_width = frame_width * self.config.plotting.gridLayout[0]
        grid_height = frame_height * self.config.plotting.gridLayout[1]
        if self.config.plotting.outputFrameHeight > 0:
            grid_scale = self.config.plotting.outputFrameHeight / float(grid_height)
        else:
            self.config.plotting.outputFrameHeight = grid_height
            grid_scale = 1.
        video_writer = cv2.VideoWriter(output_video_name,
                                       cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                       fps, (int(grid_width * grid_scale),
                                             self.config.plotting.outputFrameHeight))

        # Initialize input video captures and tails
        video_captures: Dict[str, cv2.VideoCapture] = dict()
        tails: Dict[str, Dict[str, deque]] = dict()
        map_sensor_id_to_behavior_keys: Dict[str, Set[str]] = dict()
        for sensor_id in sensor_ids:
            input_video_path = os.path.join(self.config.io.videoDirPath, f"{sensor_id}.mp4")
            video_capture = cv2.VideoCapture(input_video_path)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, global_start_frame)
            video_captures[sensor_id] = video_capture
            tails[sensor_id] = dict()
            map_sensor_id_to_behavior_keys[sensor_id] = set()

        bbox_color = get_predefined_bgr_color(global_idx)
        for frame_id in range(global_start_frame, global_end_frame + 1):
            print(f"Plotting MTMC object in grid @ global_id {global_id}; frame_id {frame_id:06}")

            image_grid = np.zeros((frame_height * self.config.plotting.gridLayout[1], frame_width * self.config.plotting.gridLayout[0], 3), np.uint8)
            frame_info = frames.get(str(frame_id), None)

            for sensor_idx in range(len(sensor_ids)):
                sensor_id = sensor_ids[sensor_idx]
                success, image_frame = video_captures[sensor_id].read()
                if not success:
                    continue

                plot_frame = False
                if (frame_info is not None) and (sensor_id in frame_info.keys()):
                    # Get bbox of each sensor
                    behavior_keys: List[str] = list()
                    bbox_object_ids: List[str] = list()
                    bboxes: List[Bbox] = list()
                    foot_pixels: List[List[int]] = list()
                    convex_hulls: List[Optional[List[int]]] = list()
                    for object_id in frame_info[sensor_id].keys():
                        # Get behavior keys
                        for behavior_key in mtmc_objects[global_id].keys():
                            behavior_tokens = behavior_key.split(" #-# ")
                            if (sensor_id == behavior_tokens[0]) and (object_id == behavior_tokens[1]):
                                if (frame_id >= int(behaviors[sensor_id][behavior_key].startFrame)) and \
                                    (frame_id <= int(behaviors[sensor_id][behavior_key].endFrame)):
                                    try:
                                        idx = behaviors[sensor_id][behavior_key].frameIds.index(str(frame_id))
                                    except ValueError:
                                        continue
                                    map_sensor_id_to_behavior_keys[sensor_id].add(behavior_key)
                                    behavior_keys.append(behavior_key)
                                    bbox_object_ids.append(object_id)
                                    bbox = behaviors[sensor_id][behavior_key].bboxes[idx]
                                    bboxes.append(bbox)
                                    if behavior_key not in tails[sensor_id].keys():
                                        tails[sensor_id][behavior_key] = deque(list())
                                    object_instance = frames[str(frame_id)][sensor_id][object_id]

                                    convex_hull = None
                                    if (object_instance.info is not None) and ("convexHull" in object_instance.info.keys()):
                                        try:
                                            convex_hull = object_instance.info["convexHull"].split(",")
                                        except ValueError:
                                            pass
                                    convex_hulls.append(convex_hull)

                                    foot_pixel = None
                                    if (object_instance.info is not None) and ("footLocation" in object_instance.info.keys()):
                                        try:
                                            foot_tokens = object_instance.info["footLocation"].split(",")
                                            foot_pixel = [round(float(foot_tokens[0])), round(float(foot_tokens[1]))]
                                            del foot_tokens
                                        except ValueError:
                                            pass
                                    if foot_pixel is None:
                                        foot_pixel = [round((bbox.leftX + bbox.rightX) / 2.), round(bbox.bottomY)]
                                    foot_pixels.append(foot_pixel)

                    # if len(bboxes) > 1:
                    #     logging.error(f"ERROR: Multiple objects are matched to the same MTMC object at the frame {frame_id}.")
                    #     exit(1)

                    for behavior_key in tails[sensor_id].keys():
                        if behavior_key not in behavior_keys:
                            tails[sensor_id][behavior_key].clear()

                    for i in range(len(bboxes)):
                        plot_frame = True
                        bbox = bboxes[i]
                        foot_pixel = foot_pixels[i]
                        convex_hull = convex_hulls[i]
                        behavior_key = behavior_keys[i]

                        # Update tail
                        while len(tails[sensor_id][behavior_key]) >= self.config.plotting.tailLengthMax:
                            tails[sensor_id][behavior_key].popleft()
                        tails[sensor_id][behavior_key].append(foot_pixel)

                        # Smooth tail
                        if len(tails[sensor_id][behavior_key]) > self.config.plotting.smoothingTailLengthThresh:
                            tail_smoothed = smooth_tail(tails[sensor_id][behavior_key], self.config.plotting.smoothingTailWindow)
                        else:
                            tail_smoothed = tails[sensor_id][behavior_key]

                        # Plot the tail
                        tail_smoothed.append(foot_pixel)
                        for t in range(len(tail_smoothed) - 1):
                            cv2.line(image_frame, tail_smoothed[t], tail_smoothed[t+1], bbox_color, 3)

                        # Plot the object
                        bbox_label = f"{global_id}"
                        image_frame = plot_object(image_frame, bbox, convex_hull, foot_pixel, bbox_label, bbox_color)
                        del bbox
                        del convex_hull
                        del foot_pixel

                else:
                    for behavior_key in tails[sensor_id].keys():
                        tails[sensor_id][behavior_key].clear()

                # Check whether the current behavior has ended
                if not plot_frame:
                    for behavior_key in map_sensor_id_to_behavior_keys[sensor_id]:
                        behavior = behaviors[sensor_id][behavior_key]
                        if int(behavior.startFrame) <= frame_id <= int(behavior.endFrame):
                            plot_frame = True
                            break

                if (not self.config.plotting.blankOutEmptyFrames) or plot_frame:
                    # Plot the frame ID
                    frame_label = ("Cam %d - " % (sensor_idx + 1)) + ("Frame No. %06d" % frame_id)
                    image_frame = plot_frame_label(image_frame, frame_label)

                    # Dim images without object
                    if not plot_frame:
                        image_frame = cv2.convertScaleAbs(image_frame, alpha=DIMMING_FACTOR)

                    # Assign grid location
                    idx_sensor = sensor_ids.index(sensor_id)
                    idx_column = idx_sensor % self.config.plotting.gridLayout[0]
                    idx_row = int(idx_sensor / self.config.plotting.gridLayout[0])
                    image_grid[idx_row * frame_height : (idx_row + 1) * frame_height,
                               idx_column * frame_width : (idx_column + 1) * frame_width, :] = image_frame

            # Plot white edges for the grid
            for idx_column in range(1, self.config.plotting.gridLayout[0]):
                x_edge = idx_column * frame_width
                cv2.line(image_grid, (x_edge, 0), (x_edge, grid_height), (255, 255, 255), 4)
            for idx_row in range(1, self.config.plotting.gridLayout[1]):
                y_edge = idx_row * frame_height
                cv2.line(image_grid, (0, y_edge), (grid_width, y_edge), (255, 255, 255), 4)

            image_grid = cv2.resize(image_grid, (int(grid_width * grid_scale), self.config.plotting.outputFrameHeight))
            video_writer.write(image_grid)
            del image_grid

        # Release video captures
        for sensor_id in sensor_ids:
            video_captures[sensor_id].release()

        # Release video writer
        video_writer.release()

        # Convert MPEG4 video into H.264
        if self.config.setup.ffmpegRequired:
            h264_video_name = f"object_id_{global_id}.mp4"
            stream = ffmpeg.input(output_video_name)
            stream = ffmpeg.output(stream, h264_video_name)
            ffmpeg.run(stream)
            h264_video_path = os.path.join(output_video_dir, h264_video_name)
            shutil.move(h264_video_name, h264_video_path)
            if os.path.isfile(output_video_name):
                os.remove(output_video_name)
            logging.info(f"Saved plotted video at {validate_file_path(h264_video_path)}")
        else:
            output_video_path = os.path.join(output_video_dir, output_video_name)
            shutil.move(output_video_name, output_video_path)
            logging.info(f"Saved plotted video at {validate_file_path(output_video_path)}")
