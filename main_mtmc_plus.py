# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import argparse
import os
import logging
import multiprocessing
import math
from typing import List, Dict

from mdx.mtmc.schema import Behavior, MTMCObject
from mdx.mtmc.config import AppConfig
from mdx.mtmc.core.calibration import Calibrator
from mdx.mtmc.core.data import Preprocessor
from mdx.mtmc.core.clustering import Clusterer
from mdx.mtmc.stream.config_manager import ConfigManager
from mdx.mtmc.stream.data_transformer import DataTransformer
from mdx.mtmc.stream.kafka_message_broker import KafkaMessageBroker
from mdx.mtmc.stream.multiprocessor import MultiProcessor
from mdx.mtmc.stream.state.behavior import StateManager
from mdx.mtmc.stream.state.mtmc_plus import StateManager as MTMCPlusStateManager
from mdx.mtmc.stream.state.sensor import StateManager as SensorStateManager
from mdx.mtmc.stream.state.people_height import StateManager as PeopleHeightEstimator
from mdx.mtmc.utils.io_utils import ValidateFile, validate_file_path, load_json_from_file

logging.basicConfig(format="%(asctime)s.%(msecs)03d - %(message)s",
                    datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)


class RTLSStreamProcessingApp:
    """
    Controller module for RTLS stream processing

    :param str config_path: path to the app config file
    :param str calibration_path: path to the calibration file in JSON format
    ::

        stream_processing_app = RTLSStreamProcessingApp(config_path, calibration_path)
    """

    def __init__(self, config_path: str, calibration_path: str) -> None:
        # Make sure the config file exists
        valid_config_path = validate_file_path(config_path)
        if not os.path.exists(valid_config_path):
            logging.error(
                f"ERROR: The indicated config file `{valid_config_path}` does NOT exist.")
            exit(1)

        self.config = AppConfig(**load_json_from_file(config_path))
        logging.info(f"Read config from {valid_config_path}\n")
        
        if self.config.streaming.mtmcPlusBehaviorBatchesConsumed < 1:
            logging.error(f"ERROR: config.streaming.mtmcPlusBehaviorBatchesConsumed should have a minimum value of 1.")
            exit(1)

        self._batch_id = 0
        self.calibration_path = calibration_path
        self.kafka_message_broker = KafkaMessageBroker(self.config)
        self.calibrator = Calibrator(self.config)
        self.config_manager = ConfigManager(self.config,self.kafka_message_broker)
        self.data_transformer = DataTransformer()
        self.data_preprocessor = Preprocessor(self.config)
        self.behavior_state = StateManager(self.config)
        self.clusterer = Clusterer(self.config)
        self.mtmc_plus_state = MTMCPlusStateManager(self.config)
        self.sensor_state = SensorStateManager()
        self.multi_processor = MultiProcessor(self.config, self.data_transformer, self.data_preprocessor, self.kafka_message_broker)

    def _get_behaviors_from_shared_behavior_lists(self, shared_behavior_lists: List[List[Behavior]], mtmc_plus_state_update_next_batch: bool) -> List[Behavior]:
        """
        Gets behavior list from shared behavior lists

        :param List[List[Behavior]] shared_behavior_lists: list of multiple Manager.list objects containing behavior data
        :param bool mtmc_plus_state_update_next_batch: flag indicating if MTMC plus state will be updated in the upcoming batch
        :return: list of behaviors
        :rtype: List[Behavior]
        """
        behavior_list: List[Behavior] = list()
        behavior_batches_to_be_consumed = self.config.streaming.mtmcPlusBehaviorBatchesConsumed
        batches_from_beginning = math.floor(behavior_batches_to_be_consumed / 2)
        batches_from_end = math.ceil(behavior_batches_to_be_consumed / 2)
        for behavior_shared_list in shared_behavior_lists:
            current_length = len(behavior_shared_list)
            if current_length > 0:
                retrieved_lists = list()
                if mtmc_plus_state_update_next_batch:
                    if current_length <= behavior_batches_to_be_consumed:
                        retrieved_lists = behavior_shared_list[:current_length]
                    else:
                        for retrieved_list in behavior_shared_list[:batches_from_beginning]:
                            retrieved_lists.append(retrieved_list)
                        for retrieved_list in behavior_shared_list[current_length-batches_from_end:current_length]:
                            retrieved_lists.append(retrieved_list)
                else:
                    retrieved_lists = behavior_shared_list[:current_length]

                for retrieved_list in retrieved_lists:
                    behavior_list.extend(retrieved_list)

                del behavior_shared_list[:current_length]

        return behavior_list

    def start_stream_processing(self) -> None:
        """
        Runs RTLS stream processing

        :return: None
        ::

            stream_processing_app.start_stream_processing()
        """
        # Instantiate modules
        producer = self.kafka_message_broker.get_producer()

        # Initialize sensor state
        self.sensor_state.init_state(self.calibrator, self.calibration_path)
        sensor_state_objects = self.sensor_state.get_sensor_state_objects()
        self.data_preprocessor.set_sensor_state_objects(sensor_state_objects)
        self.mtmc_plus_state.set_sensor_state_objects(sensor_state_objects)

        # Initialize estimator of people's height
        self.people_height_estimator = PeopleHeightEstimator(self.config)
        self.data_preprocessor.set_people_height_estimator(self.people_height_estimator)

        del sensor_state_objects

        shared_behavior_lists: List[List[Behavior]] = list()
        manager = multiprocessing.Manager()
        processes: List[multiprocessing.Process] = list()
        for i in range(self.config.streaming.mtmcPlusNumProcessesMax):
            behavior_shared_list = manager.list()
            process = multiprocessing.Process(target=self.multi_processor.consume_raw_messages_and_add_to_behavior_list, args=(i, behavior_shared_list))
            shared_behavior_lists.append(behavior_shared_list)
            processes.append(process)
            process.start()

        mtmc_plus_state_update_next_batch = False
        try:
            while True:
                # Obtain behavior lists from shared lists
                batch_behaviors = self._get_behaviors_from_shared_behavior_lists(shared_behavior_lists, mtmc_plus_state_update_next_batch)
                if len(batch_behaviors) == 0:
                    if self._batch_id > 0:
                        self._batch_id += 1
                        deleted_behavior_keys = self.behavior_state.delete_older_state()
                        self.mtmc_plus_state.delete_older_state(deleted_behavior_keys)
                        logging.debug(f"Empty list for batch: {self._batch_id}")
                else:
                    self._batch_id += 1
                    logging.info(f"Processing batch: {self._batch_id}")
                    logging.info(f"Created a total of {len(batch_behaviors)} behavior(s) in batch: {self._batch_id}...")

                    # Update people height
                    self.data_preprocessor.update_people_height(batch_behaviors)

                    # Update behavior state
                    self.behavior_state.update_state(batch_behaviors, self.data_preprocessor)
                    logging.info(f"Updated behavior state...")

                    # Get all live behaviors from state
                    live_behaviors = self.behavior_state.get_behaviors_in_state()
                    logging.info(f"Got {len(live_behaviors)} live behavior(s) from state...")

                    # Delay timestamps of behaviors
                    live_behaviors = self.data_preprocessor.delay_behavior_timestamps(live_behaviors)

                    # Filter live behaviors
                    live_behaviors = self.data_preprocessor.filter(live_behaviors)
                    logging.info(f"Filtered live behavior(s)...")

                    # Normalize embeddings of behaviors
                    for i in range(len(live_behaviors)):
                        live_behaviors[i].embeddings = self.data_preprocessor.normalize_embeddings(live_behaviors[i].embeddings)
                    logging.info(f"Normalized embedding(s) of live behavior(s)...")

                    # Group live behaviors by places
                    map_place_to_live_behaviors = self.data_preprocessor.group_behaviors_by_places(live_behaviors)
                    del live_behaviors

                    for place in map_place_to_live_behaviors.keys():
                        mtmc_objects: List[MTMCObject] = list()
                        map_global_id_to_mean_embedding: Dict[str, List[List[float]]] = dict()

                        # Initialize MTMC plus state
                        init_mtmc_plus_state = False
                        overwritten_num_clusters = self.config.clustering.overwrittenNumClusters
                        mtmc_state_objects_plus = self.mtmc_plus_state.get_mtmc_state_objects_plus()
                        if self.mtmc_plus_state.check_init_state(map_place_to_live_behaviors[place]):
                            init_mtmc_plus_state = True
                            mtmc_plus_state_update_next_batch = False

                            if (overwritten_num_clusters is not None) and (len(map_place_to_live_behaviors[place]) < overwritten_num_clusters):
                                logging.info(f"Skipped clustering because the number of live behavior(s) {len(map_place_to_live_behaviors[place])} "
                                             f"is smaller than the expected number of cluster(s) {overwritten_num_clusters}...")
                                continue

                            if not self.mtmc_plus_state.check_init_buffer_ready(map_place_to_live_behaviors[place]):
                                logging.info(f"Skipped clustering because the buffer of live behaviors is too short...")
                                continue

                            mtmc_plus_state_update_next_batch = True

                            # Cluster the behaviors to get MTMC objects
                            mtmc_objects, behavior_keys, embedding_array = self.clusterer.cluster(map_place_to_live_behaviors[place])
                            logging.info(f"Clustered live behavior(s) to get MTMC object(s)...")

                            # if (overwritten_num_clusters is not None) and (len(mtmc_objects) < overwritten_num_clusters):
                            #     logging.info(f"Skipped clustering because the number of output cluster(s) {len(mtmc_objects)} "
                            #                  f"is smaller than the expected number of cluster(s) {overwritten_num_clusters}...")
                            #     continue

                            # Stitch MTMC objects with the state
                            if len(mtmc_state_objects_plus) > 0:
                                mtmc_objects = self.clusterer.stitch_mtmc_objects_with_state(mtmc_objects, mtmc_state_objects_plus)
                                self.mtmc_plus_state.clean_unused_state(mtmc_objects)
                                logging.info(f"Stitched MTMC objects with the state...")

                            # Compute mean embeddings
                            map_global_id_to_mean_embedding = self.mtmc_plus_state.compute_mean_embeddings(mtmc_objects, behavior_keys, embedding_array)
                            logging.info(f"Computed mean embedding(s) of MTMC object(s)...")

                            del behavior_keys
                            del embedding_array

                        else:
                            # Match the behaviors to get MTMC objects in online mode
                            mtmc_objects, map_global_id_to_mean_embedding = self.clusterer.match_online(map_place_to_live_behaviors[place], mtmc_state_objects_plus)
                            logging.info(f"Matched live behavior(s) in online mode to get MTMC object(s)...")

                            # Update mean embeddings
                            map_global_id_to_mean_embedding = self.mtmc_plus_state.update_mean_embeddings(map_global_id_to_mean_embedding)
                            logging.info(f"Updated mean embedding(s) of MTMC object(s)...")

                        # Get MTMC objects plus locations
                        mtmc_objects_plus = self.mtmc_plus_state.get_mtmc_objects_plus(mtmc_objects, map_place_to_live_behaviors[place])
                        del mtmc_objects
                        logging.info(f"Got MTMC object(s) plus location(s)...")

                        # Update MTMC objects plus locations
                        updated_mtmc_objects_plus = self.mtmc_plus_state.update_mtmc_objects_plus(mtmc_objects_plus, map_global_id_to_mean_embedding)
                        del mtmc_objects_plus
                        del map_global_id_to_mean_embedding
                        logging.info(f"Updated MTMC object(s) plus location(s)...")

                        if not init_mtmc_plus_state:
                            # Produce MTMC plus messages
                            self.kafka_message_broker.produce_mtmc_plus_messages(producer, updated_mtmc_objects_plus)
                            del updated_mtmc_objects_plus

                            # Delete older state
                            deleted_behavior_keys = self.behavior_state.delete_older_state()
                            self.mtmc_plus_state.delete_older_state(deleted_behavior_keys)

                    del map_place_to_live_behaviors

                    logging.info(f"Processed batch: {self._batch_id}")

        except (KeyboardInterrupt, SystemExit):
            logging.info("Main process interrupted. Stopping...")

        finally:
            producer.close()
            for process in processes:
                process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=validate_file_path, default="resources/app_rtls_config.json",
                        action=ValidateFile, help="The input app config file")
    parser.add_argument("--calibration", type=validate_file_path, default="resources/calibration_retail_synthetic.json",
                        action=ValidateFile, help="The input calibration file")
    args = parser.parse_args()
    stream_processing_app = RTLSStreamProcessingApp(args.config, args.calibration)
    stream_processing_app.start_stream_processing()
