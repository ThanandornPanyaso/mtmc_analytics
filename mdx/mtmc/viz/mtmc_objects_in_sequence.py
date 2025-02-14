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
from mdx.mtmc.utils.io_utils import validate_file_path
from mdx.mtmc.utils.viz_mtmc_utils import get_predefined_bgr_color, smooth_tail, \
    plot_object, plot_frame_label


class VizMTMCObjectsInSequence:
    """
    Module to visualize MTMC objects in sequence

    :param VizMtmcConfig config: configuration for visualization
    ::

        visualizer = VizMTMCObjectsInSequence(config)
    """

    def __init__(self, config: VizMtmcConfig) -> None:
        self.config: VizMtmcConfig = config

    def plot(self, global_id: str,
             output_video_dir: str,
             frames: Dict[str, Dict[str, Dict[str, Object]]],
             behaviors: Dict[str, Dict[str, Behavior]],
             mtmc_objects: Dict[str, Dict[str, Behavior]]) -> None:
        """
        Visualizes MTMC objects in sequence

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
        if self.config.plotting.outputFrameHeight > 0:
            frame_scale = self.config.plotting.outputFrameHeight / float(frame_height)
        else:
            self.config.plotting.outputFrameHeight = frame_height
            frame_scale = 1.
        video_writer = cv2.VideoWriter(output_video_name,
                                        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                        fps, (int(frame_width * frame_scale),
                                              self.config.plotting.outputFrameHeight))

        # Start drawing trajectory
        sensor_ids = sorted(list(behaviors))
        bbox_color = get_predefined_bgr_color(global_idx)
        for behavior_key in mtmc_objects[global_id].keys():
            matched_behavior = mtmc_objects[global_id][behavior_key].copy()
            sensor_id = matched_behavior.sensorId
            sensor_idx = sensor_ids.index(sensor_id)
            object_id = matched_behavior.objectId
            matched_behavior_key = matched_behavior.key
            bboxes = behaviors[sensor_id][matched_behavior_key].bboxes
            frame_ids = behaviors[sensor_id][matched_behavior_key].frameIds

            # Determine local starting and ending frames
            start_frame = int(matched_behavior.startFrame)
            end_frame = int(matched_behavior.endFrame)
            behavior_length = end_frame - start_frame + 1

            # Locate the starting frame
            input_video_path = os.path.join(self.config.io.videoDirPath, f"{sensor_id}.mp4")
            video_capture = cv2.VideoCapture(input_video_path)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            tail = deque(list())
            for f in range(behavior_length):
                frame_id = start_frame + f
                print(f"Plotting MTMC object in sequence @ global_id {global_id}; behavior_key {behavior_key}; frame_id {frame_id:06}")

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
                            bbox_label = f"{global_id}"
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
