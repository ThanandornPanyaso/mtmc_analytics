# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import shutil
import logging
import cv2
import ffmpeg
from typing import Dict
from collections import deque

from mdx.mtmc.config import VizMtmcConfig
from mdx.mtmc.schema import Object, Behavior
from mdx.mtmc.utils.io_utils import validate_file_path, sanitize_string
from mdx.mtmc.utils.viz_mtmc_utils import get_random_bgr_color, smooth_tail, \
    plot_object, plot_frame_label


class VizBehaviors:
    """
    Module to visualize behaviors

    :param VizMtmcConfig config: configuration for visualization
    ::

        visualizer = VizBehaviors(config)
    """

    def __init__(self, config: VizMtmcConfig) -> None:
        self.config: VizMtmcConfig = config

    def plot(self, sensor_id: str,
             output_video_dir: str,
             frames: Dict[str, Dict[str, Dict[str, Object]]],
             behaviors: Dict[str, Dict[str, Behavior]]) -> None:
        """
        Visualizes behaviors

        :param str sensor_id: sensor ID
        :param str output_video_dir: output directory for videos
        :param Dict[str,Dict[str,Dict[str,Object]]] frames: frames extracted from raw data
        :param Dict[str,Dict[str,Behavior]] behaviors: behaviors, i.e., tracklets
        :return: None
        ::

            visualizer.plot(sensor_id, output_video_dir, frames, behaviors)
        """
        # Get the index of the sensor ID
        sensor_ids = sorted(list(frames))
        sensor_idx = sensor_ids.index(sensor_id)

        input_video_path = os.path.join(self.config.io.videoDirPath, f"{sensor_id}.mp4")
        if not os.path.exists(input_video_path):
            logging.error(f"ERROR: The input video path `{input_video_path}` does NOT exist.")
            exit(1)

        for behavior_key in behaviors[sensor_id].keys():
            start_frame = int(behaviors[sensor_id][behavior_key].startFrame)
            end_frame = int(behaviors[sensor_id][behavior_key].endFrame)
            behavior_length = end_frame - start_frame + 1

            bboxes = behaviors[sensor_id][behavior_key].bboxes
            frame_ids = behaviors[sensor_id][behavior_key].frameIds
            object_id = behaviors[sensor_id][behavior_key].objectId
            bbox_color = get_random_bgr_color(int(object_id))

            # Initialize video capture
            video_capture = cv2.VideoCapture(input_video_path)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Initialize video writer
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            if self.config.setup.ffmpegRequired:
                output_video_name = sanitize_string(f"behavior_key_{behavior_key}_mpeg4.mp4")
            else:
                output_video_name = sanitize_string(f"behavior_key_{behavior_key}.mp4")
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

            tail = deque(list())
            for f in range(behavior_length):
                frame_id = start_frame + f
                print(f"Plotting behavior @ behavior_key {behavior_key}; frame_id {frame_id:06}")

                try:
                    bbox = bboxes[frame_ids.index(str(frame_id))]
                except ValueError:
                    bbox = None

                success, image_frame = video_capture.read()
                if not success:
                    break

                if bbox is not None:
                    # Update tail
                    object_instance = frames[sensor_id][str(frame_id)][object_id]

                    convex_hull = None
                    if (object_instance.info is not None) and ("convexHull" in object_instance.info.keys()):
                        try:
                            convex_hull = object_instance.info["convexHull"].split(",")
                        except ValueError:
                            pass

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

                    while len(tail) >= self.config.plotting.tailLengthMax:
                        tail.popleft()
                    tail.append(foot_pixel)

                    if str(frame_id) in frames[sensor_id]:
                        object_ids = list(frames[sensor_id][str(frame_id)])

                        if object_id in object_ids:
                            # Smooth tail
                            if len(tail) > self.config.plotting.smoothingTailLengthThresh:
                                tail_smoothed = smooth_tail(tail, self.config.plotting.smoothingTailWindow)
                            else:
                                tail_smoothed = tail

                            # Plot the tail
                            tail_smoothed.append(foot_pixel)
                            for t in range(len(tail_smoothed) - 1):
                                cv2.line(image_frame, tail_smoothed[t], tail_smoothed[t+1], bbox_color, 3)

                            # Plot the object
                            bbox_label = f"{object_id}"
                            image_frame = plot_object(image_frame, bbox, convex_hull, foot_pixel, bbox_label, bbox_color)
                            del bbox
                            del convex_hull
                            del foot_pixel

                        else:
                            tail.clear()

                    else:
                        tail.clear()

                # Plot the sensor ID and frame ID
                frame_label = ("Cam %d - " % (sensor_idx + 1)) + ("Frame No. %06d" % frame_id)
                image_frame = plot_frame_label(image_frame, frame_label)

                image_frame = cv2.resize(image_frame, (int(frame_width * frame_scale), self.config.plotting.outputFrameHeight))
                video_writer.write(image_frame)

            # Release video capture
            video_capture.release()

            # Release video writer
            video_writer.release()

            # Convert MPEG4 video into H.264
            if self.config.setup.ffmpegRequired:
                h264_video_name = sanitize_string(f"behavior_key_{behavior_key}.mp4")
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
