# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import shutil
import logging
import cv2
import ffmpeg
from typing import List, Dict, Tuple
from collections import OrderedDict, defaultdict, deque

from mdx.mtmc.config import VizMtmcConfig
from mdx.mtmc.core.calibration import Calibrator
from mdx.mtmc.utils.io_utils import validate_file_path
from mdx.mtmc.utils.viz_mtmc_utils import get_random_bgr_color, smooth_tail, plot_frame_label


class VizGroundTruthLocations:
    """
    Module to visualize ground-truth locations in top view

    :param dict config: configuration for visualization
    ::

        visualizer = VizGroundTruthLocations(config)
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

    def plot(self, output_video_dir: str, ground_truth_locations: Dict[str, Dict[str, List[float]]], max_object_id: int) -> None:
        """
        Visualizes ground-truth locations in top view

        :param str output_video_dir: output directory for videos
        :param Dict[str,Dict[str,List[float]]] ground_truth_locations: ground-truth locations
        :param int max_object_id: max object ID
        :return: None
        ::

            visualizer.plot(output_video_dir, ground_truth_locations, max_object_id)
        """
        # Read map image
        if not os.path.isfile(self.config.io.mapPath):
            logging.error(f"ERROR: The map image {self.config.io.mapPath} for top-view visualization does NOT exist.")
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
        input_video_path = os.path.join(self.config.io.videoDirPath, os.listdir(self.config.io.videoDirPath)[0])
        if not os.path.exists(input_video_path):
            logging.error(f"ERROR: The input video path `{input_video_path}` does NOT exist.")
            exit(1)
        video_capture = cv2.VideoCapture(input_video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        if self.config.setup.ffmpegRequired:
            output_video_name = f"ground_truth_locations_mpeg4.mp4"
        else:
            output_video_name = f"ground_truth_locations.mp4"
        video_writer = cv2.VideoWriter(output_video_name,
                                       cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                       fps, (int(map_width * frame_scale),
                                             self.config.plotting.outputFrameHeight))

        # Set global starting and ending frames
        first_frame_id = list(ground_truth_locations)[0]
        global_start_frame = int(first_frame_id)
        global_end_frame = int(first_frame_id)
        for frame_id in ground_truth_locations.keys():
            if int(frame_id) < global_start_frame:
                global_start_frame = int(frame_id)
            if int(frame_id) > global_end_frame:
                global_end_frame = int(frame_id)

        # Load objects from frames
        tails: Dict[str, Dict[int, List[int]]] = dict()
        for frame_id in ground_truth_locations:
            for object_id in ground_truth_locations[frame_id].keys():
                if object_id not in tails:
                    tails[object_id] = OrderedDict()
                location = ground_truth_locations[frame_id][object_id]
                location[0] = int((location[0] + self.translation_to_global_coordinates["x"]) * self.scale_factor * frame_scale)
                location[1] = int(self.config.plotting.outputFrameHeight - ((location[1] + self.translation_to_global_coordinates["y"]) * self.scale_factor * frame_scale) - 1.0)
                tails[object_id][int(frame_id)] = location

        # Set plotting config
        map_object_id_to_color: Dict[str, Tuple[int, int, int]] = dict()
        map_object_id_to_tail_length: Dict[str, int] = defaultdict(int)
        text_face = cv2.FONT_HERSHEY_DUPLEX
        text_scale = 1.0
        text_thickness = 2
        circle_size, _ = cv2.getTextSize(str(max_object_id), text_face, text_scale, text_thickness)

        for frame_id in range(global_start_frame, global_end_frame + 1):
            print(f"Plotting ground-truth locations @ frame_id {frame_id:06}")
            image_frame = image_map.copy()
            frame_label = "Frame No. %06d" % frame_id

            if str(frame_id) in ground_truth_locations.keys():
                for object_id in ground_truth_locations[str(frame_id)].keys():
                    location = ground_truth_locations[str(frame_id)][object_id]

                    # Set location color
                    location_color = None
                    if object_id in map_object_id_to_color:
                        location_color = map_object_id_to_color[object_id]
                    else:
                        location_color = get_random_bgr_color(int(object_id))
                        map_object_id_to_color[object_id] = location_color

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
                        cv2.line(image_frame, tail_smoothed[-(t+1)], tail_smoothed[-(t+2)], map_object_id_to_color[object_id], 2)

                    # Plot the current location
                    cv2.circle(image_frame, tuple(tail_smoothed[-1]), int(max(circle_size) * 0.75), location_color, -1)
                    text_size, _ = cv2.getTextSize(object_id, text_face, text_scale, text_thickness)
                    text_location = (int((tail_smoothed[-1][0]) - text_size[0] / 2),
                                     int((tail_smoothed[-1][1]) + text_size[1] / 2))
                    cv2.putText(image_frame, object_id, text_location, text_face,
                                text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

            else:
                # Reset tail lengths
                for object_id in tails.keys():
                    map_object_id_to_tail_length[object_id] = 0

            # Plot the frame ID
            image_frame = plot_frame_label(image_frame, frame_label)

            video_writer.write(image_frame)

        # Release video writer
        video_writer.release()

        # Convert MPEG4 video into H.264
        if self.config.setup.ffmpegRequired:
            h264_video_name = f"ground_truth_locations.mp4"
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
