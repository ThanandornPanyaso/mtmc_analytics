# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import shutil
import logging
import cv2
import ffmpeg
import numpy as np
from typing import List, Dict
from collections import deque

from mdx.mtmc.config import VizMtmcConfig
from mdx.mtmc.schema import Object, Behavior
from mdx.mtmc.core.calibration import Calibrator
from mdx.mtmc.utils.io_utils import validate_file_path
from mdx.mtmc.utils.viz_mtmc_utils import get_predefined_bgr_color, smooth_tail, plot_frame_label


class VizMTMCObjectsInTopview:
    """
    Module to visualize MTMC objects in top view

    :param dict config: configuration for visualization
    ::

        visualizer = VizMTMCObjectsInTopview(config)
    """

    def __init__(self, config: VizMtmcConfig, calibration_path: str) -> None:
        self.config: VizMtmcConfig = config
        calibrator = Calibrator(None)
        sensors = calibrator.load_calibration_file(calibration_path)
        self.scale_factor = sensors[0].scaleFactor
        self.translation_to_global_coordinates = sensors[0].translationToGlobalCoordinates
        for sensor in sensors:
            if sensor.scaleFactor != self.scale_factor:
                logging.error(f"ERROR: The scale factors of sensors are inconsistent: {sensor.scaleFactor} != {self.scale_factor}.")
                exit(1)
            if sensor.translationToGlobalCoordinates != self.translation_to_global_coordinates:
                logging.error(f"ERROR: The translation vectors to global coordinates of sensors are inconsistent: "
                              f"{sensor.translationToGlobalCoordinates} != {self.translation_to_global_coordinates}.")
                exit(1)

    def plot(self, global_id: str,
             output_video_dir: str,
             frames: Dict[str, Dict[str, Dict[str, Object]]],
             behaviors: Dict[str, Dict[str, Behavior]],
             mtmc_objects: Dict[str, Dict[str, Behavior]]) -> None:
        """
        Visualizes MTMC objects in top view

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

        # Read map image
        if not os.path.isfile(self.config.io.mapPath):
            logging.error("ERROR: The map image for top-view visualization does NOT exist.")
            exit(1)
        image_map = cv2.imread(self.config.io.mapPath)
        map_height, map_width = image_map.shape[:2]
        if self.config.plotting.outputFrameHeight > 0:
            frame_scale = self.config.plotting.outputFrameHeight / float(map_height)
        else:
            self.config.plotting.outputFrameHeight = map_height
            frame_scale = 1.
        image_map = cv2.resize(image_map, (int(map_width * frame_scale), self.config.plotting.outputFrameHeight))

        # Initialize video writer
        input_video_path = os.path.join(self.config.io.videoDirPath, f"{list(behaviors)[0]}.mp4")
        if not os.path.exists(input_video_path):
            logging.error(f"ERROR: The input video path `{input_video_path}` does NOT exist.")
            exit(1)
        video_capture = cv2.VideoCapture(input_video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        if self.config.setup.ffmpegRequired:
            output_video_name = f"object_id_{global_id}_mpeg4.mp4"
        else:
            output_video_name = f"object_id_{global_id}.mp4"
        video_writer = cv2.VideoWriter(output_video_name,
                                       cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                       fps, (int(map_width * frame_scale),
                                             self.config.plotting.outputFrameHeight))

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

        # Set plotting config
        location_color = get_predefined_bgr_color(global_idx)
        text_face = cv2.FONT_HERSHEY_DUPLEX
        text_scale = 2.0
        text_thickness = 2
        label_size, _ = cv2.getTextSize(global_id, text_face, text_scale, text_thickness)

        tail = deque(list())
        for frame_id in range(global_start_frame, global_end_frame + 1):
            print(f"Plotting MTMC object in top view @ global_id {global_id}; frame_id {frame_id:06}")

            image_frame = image_map.copy()
            frame_info = frames.get(str(frame_id), None)
            frame_label = "Frame No. %06d" % frame_id

            if frame_info is not None:
                locations: List[List[float]] = list()
                for sensor_id in frame_info.keys():
                    # Aggregate locations at a frame for the given global ID
                    # num_matched_objects = 0
                    for object_id in frame_info[sensor_id].keys():
                        # Get behavior key
                        behavior_key = None
                        for behavior_key_candidate in mtmc_objects[global_id].keys():
                            behavior_tokens = behavior_key_candidate.split(" #-# ")
                            if (sensor_id == behavior_tokens[0]) and (object_id == behavior_tokens[1]):
                                if (frame_id >= int(behaviors[sensor_id][behavior_key_candidate].startFrame)) and \
                                    (frame_id <= int(behaviors[sensor_id][behavior_key_candidate].endFrame)):
                                    if str(frame_id) in behaviors[sensor_id][behavior_key_candidate].frameIds:
                                        behavior_key = behavior_key_candidate
                                        break
                        if behavior_key is None:
                            continue

                        idx = behaviors[sensor_id][behavior_key].frameIds.index(str(frame_id))
                        location_mask = behaviors[sensor_id][behavior_key].locationMask
                        if not location_mask[idx]:
                            continue
                        idx_location = 0
                        for i in range(idx):
                            if location_mask[i]:
                                idx_location += 1
                        location = behaviors[sensor_id][behavior_key].locations[idx_location]
                        locations.append([(location[0] + self.translation_to_global_coordinates["x"]) * self.scale_factor * frame_scale,
                                          self.config.plotting.outputFrameHeight - ((location[1] + self.translation_to_global_coordinates["y"]) * self.scale_factor * frame_scale) - 1.0])
                        # num_matched_objects += 1

                    # # Raise error if multiple objects are matched to the same MTMC object at the same frame
                    # if num_matched_objects > 1:
                    #     logging.error(f"ERROR: {num_matched_objects} objects are matched to the same MTMC object at the frame {frame_id}.")
                    #     exit(1)

                # Update tail
                while len(tail) >= self.config.plotting.tailLengthMax:
                    tail.popleft()

                # Calculate the mean of locations at the same frame
                if len(locations) > 0:
                    location_indices: List[int] = list()
                    # Exclude outlier locations
                    if len(tail) > 0:
                        for i in range(len(locations)):
                            location_indices.append(i)
                    if len(location_indices) > 0:
                        location_smoothed = np.asarray([locations[i] for i in location_indices]).mean(axis=0)
                    else:
                        location_smoothed = np.asarray(locations).mean(axis=0)
                    location_smoothed = [int(location_smoothed[0]), int(location_smoothed[1])]
                    tail.append(location_smoothed)

                # Plot on the map image
                if len(tail) > 0:
                    # Smooth tail
                    if len(tail) > self.config.plotting.smoothingTailLengthThresh:
                        tail_smoothed = smooth_tail(tail, self.config.plotting.smoothingTailWindow)
                    else:
                        tail_smoothed = tail

                    # Plot the tail
                    if location_smoothed is not None:
                        tail_smoothed.append(location_smoothed)
                    for t in range(len(tail_smoothed) - 1):
                        cv2.line(image_frame, tail_smoothed[t], tail_smoothed[t+1], location_color, 4)

                    # Plot the current location
                    cv2.circle(image_frame, tuple(location_smoothed) if location_smoothed else tuple(tail[-1]),
                               int(max(label_size) * 0.75), location_color, -1)
                    text_location = (int((location_smoothed[0] if location_smoothed else tail[-1][0]) - label_size[0] / 2),
                                     int((location_smoothed[1] if location_smoothed else tail[-1][1]) + label_size[1] / 2))
                    cv2.putText(image_frame, global_id, text_location, text_face,
                                text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

            else:
                tail.clear()

            # Plot the frame ID
            image_frame = plot_frame_label(image_frame, frame_label)

            video_writer.write(image_frame)

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
