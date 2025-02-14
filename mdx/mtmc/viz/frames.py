# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import shutil
import logging
import cv2
import ffmpeg
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict, defaultdict, deque

from mdx.mtmc.config import VizMtmcConfig
from mdx.mtmc.schema import Object, Behavior
from mdx.mtmc.utils.io_utils import validate_file_path
from mdx.mtmc.utils.viz_mtmc_utils import get_random_bgr_color, smooth_tail, \
    plot_object, plot_frame_label


class VizFrames:
    """
    Module to visualize frames

    :param VizMtmcConfig config: configuration for visualization
    ::

        visualizer = VizFrames(config)
    """

    def __init__(self, config: VizMtmcConfig) -> None:
        self.config: VizMtmcConfig = config

    def plot(self, sensor_id: str,
             output_video_dir: str,
             frames: Dict[str, Dict[str, Dict[str, Object]]],
             behaviors: Optional[Dict[str, Dict[str, Behavior]]] = None) -> None:
        """
        Visualizes frames

        :param str sensor_id: sensor ID
        :param str output_video_dir: output directory for videos
        :param Dict[str,Dict[str,Dict[str,Object]]] frames: frames extracted from raw data
        :param Optional[Dict[str,Dict[str,Behavior]]] behaviors: behaviors, i.e., tracklets
        :return: None
        ::

            visualizer.plot(sensor_id, output_video_dir, frames, behaviors)
        """
        if (behaviors is not None) and (sensor_id not in behaviors.keys()):
            logging.warning(f"WARNING: All behaviors in the given sensor {sensor_id} are filtered.")
            return

        # Get the index of the sensor ID
        sensor_ids = sorted(list(frames))
        sensor_idx = sensor_ids.index(sensor_id)

        # Load objects from frames
        objects: Dict[str, Dict[str, Object]] = dict()
        tails: Dict[str, Dict[int, List[int]]] = dict()
        for frame_id in frames[sensor_id]:
            objects[frame_id] = dict()
            for object_id in frames[sensor_id][frame_id].keys():
                # Use behaviors to filter frames
                if behaviors is not None:
                    is_behavior_valid = False
                    for behavior_key in behaviors[sensor_id].keys():
                        behavior_tokens = behavior_key.split(" #-# ")
                        if (sensor_id == behavior_tokens[0]) and (object_id == behavior_tokens[1]):
                            is_behavior_valid = True
                            break
                    if not is_behavior_valid:
                        continue

                object_instance = frames[sensor_id][frame_id][object_id]
                objects[frame_id][object_id] = object_instance

                if object_id not in tails:
                    tails[object_id] = OrderedDict()
                bbox_left_x = objects[frame_id][object_id].bbox.leftX
                bbox_right_x = objects[frame_id][object_id].bbox.rightX
                bbox_bottom_y = objects[frame_id][object_id].bbox.bottomY

                foot_pixel = None
                if (object_instance.info is not None) and ("footLocation" in object_instance.info.keys()):
                    try:
                        foot_tokens = object_instance.info["footLocation"].split(",")
                        foot_pixel = [round(float(foot_tokens[0])), round(float(foot_tokens[1]))]
                        del foot_tokens
                    except ValueError:
                        pass

                if foot_pixel is None:
                    foot_pixel = [round((bbox_left_x + bbox_right_x) / 2.), round(bbox_bottom_y)]

                tails[object_id][int(frame_id)] = foot_pixel

        # Initialize video capture
        input_video_path = os.path.join(self.config.io.videoDirPath, f"{sensor_id}.mp4")
        if not os.path.exists(input_video_path):
            logging.error(f"ERROR: The input video path `{input_video_path}` does NOT exist.")
            exit(1)
        video_capture = cv2.VideoCapture(input_video_path)

        # Initialize video writer
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if self.config.setup.ffmpegRequired:
            output_video_name = f"sensor_id_{sensor_id}_mpeg4.mp4"
        else:
            output_video_name = f"sensor_id_{sensor_id}.mp4"
        if self.config.plotting.outputFrameHeight > 0:
            frame_scale = self.config.plotting.outputFrameHeight / float(frame_height)
        else:
            self.config.plotting.outputFrameHeight = frame_height
            frame_scale = 1.
        video_writer = cv2.VideoWriter(output_video_name,
                                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                        fps, (int(frame_width * frame_scale),
                                              self.config.plotting.outputFrameHeight))
        success, image_frame = video_capture.read()

        frame_id = 0
        map_object_id_to_color: Dict[str, Tuple[int, int, int]] = dict()
        map_object_id_to_tail_length: Dict[str, int] = defaultdict(int)
        while success:
            print(f"Plotting frame @ sensor_id {sensor_id}; frame_id {frame_id:06}")

            if str(frame_id) in objects.keys():
                for object_id in objects[str(frame_id)]:
                    bbox = objects[str(frame_id)][object_id].bbox

                    convex_hull = None
                    if objects[str(frame_id)][object_id].info is not None:
                        convex_hull = objects[str(frame_id)][object_id].info.get("convexHull", None)
                        if convex_hull is not None:
                            convex_hull = convex_hull.split(",")

                    # Set bbox color
                    if object_id in map_object_id_to_color:
                        bbox_color = map_object_id_to_color[object_id]
                    else:
                        bbox_color = get_random_bgr_color(int(object_id))
                        map_object_id_to_color[object_id] = bbox_color

                    # Update tail lengths
                    if frame_id in tails[object_id]:
                        map_object_id_to_tail_length[object_id] += 1
                    else:
                        # Not continuous tail
                        map_object_id_to_tail_length[object_id] = 0

                    # Smooth tail
                    tail_offset = 0
                    tail_frame_ids = sorted(tails[object_id])
                    for tail_frame_id in tail_frame_ids:
                        tail_offset += 1
                        if tail_frame_id == frame_id:
                            break
                    tail_smoothed = deque()
                    if map_object_id_to_tail_length[object_id] > 0:
                        tail_locations = [tails[object_id][tail_frame_id] for tail_frame_id in tail_frame_ids]
                        if map_object_id_to_tail_length[object_id] > self.config.plotting.smoothingTailLengthThresh:
                            tail_smoothed = smooth_tail(tail_locations[tail_offset-map_object_id_to_tail_length[object_id]:tail_offset],
                                                        self.config.plotting.smoothingTailWindow)
                        else:
                            tail_smoothed = tail_locations[tail_offset-map_object_id_to_tail_length[object_id]:tail_offset]

                    # Plot the tail
                    tail_smoothed.append(tails[object_id][frame_id])
                    for t in range(min(self.config.plotting.tailLengthMax, len(tail_smoothed) - 1)):
                        cv2.line(image_frame, tail_smoothed[-(t+1)], tail_smoothed[-(t+2)], map_object_id_to_color[object_id], 3)

                    # Plot the object
                    bbox_label = f"{object_id}"
                    image_frame = plot_object(image_frame, bbox, convex_hull, tail_smoothed[-1], bbox_label, map_object_id_to_color[object_id])
                    del bbox
                    del convex_hull

            else:
                # Reset tail lengths
                for object_id in tails.keys():
                    map_object_id_to_tail_length[object_id] = 0

            # Plot the sensor ID and frame ID
            frame_label = ("Cam %d - " % (sensor_idx + 1)) + ("Frame No. %06d" % frame_id)
            image_frame = plot_frame_label(image_frame, frame_label)
            
            image_frame = cv2.resize(image_frame, (int(frame_width * frame_scale), self.config.plotting.outputFrameHeight))
            video_writer.write(image_frame)
            success, image_frame = video_capture.read()
            frame_id += 1

        del objects
        del tails

        # Release video capture
        video_capture.release()

        # Release video writer
        video_writer.release()

        # Convert MPEG4 video into H.264
        if self.config.setup.ffmpegRequired:
            h264_video_name = f"sensor_id_{sensor_id}.mp4"
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
