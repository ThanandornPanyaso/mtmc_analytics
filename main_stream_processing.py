# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import argparse
import os
import logging
import time
from datetime import datetime, timedelta
from typing import List

from mdx.mtmc.schema import Behavior
from mdx.mtmc.config import AppConfig
from mdx.mtmc.core.calibration import Calibrator
from mdx.mtmc.core.data import Preprocessor
from mdx.mtmc.core.clustering import Clusterer
from mdx.mtmc.stream.config_manager import ConfigManager
from mdx.mtmc.stream.data_transformer import DataTransformer
from mdx.mtmc.stream.kafka_message_broker import KafkaMessageBroker
from mdx.mtmc.stream.multiprocessor import MultiProcessor
from mdx.mtmc.stream.state.behavior import StateManager
from mdx.mtmc.stream.state.mtmc import StateManager as MTMCStateManager
from mdx.mtmc.stream.state.sensor import StateManager as SensorStateManager
from mdx.mtmc.stream.state.people_height import StateManager as PeopleHeightEstimator
from mdx.mtmc.utils.io_utils import ValidateFile, validate_file_path, load_json_from_file, \
    write_pydantic_list_to_file, make_clean_dir

logging.basicConfig(format="%(asctime)s.%(msecs)03d - %(message)s",
                    datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)


class MTMCStreamProcessingApp:
    """
    Controller module for MTMC stream processing

    :param str config_path: path to the app config file
    :param str calibration_path: path to the calibration file in JSON format
    ::

        stream_processing_app = MTMCStreamProcessingApp(config_path, calibration_path)
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
        self._batch_id = 0
        self.calibration_path = calibration_path
        self.kafka_message_broker = KafkaMessageBroker(self.config)
        self.calibrator = Calibrator(self.config)
        self.config_manager = ConfigManager(self.config,self.kafka_message_broker)
        self.data_transformer = DataTransformer()
        self.data_preprocessor = Preprocessor(self.config)
        self.behavior_state = StateManager(self.config)
        self.clusterer = Clusterer(self.config)
        self.mtmc_state = MTMCStateManager()
        self.sensor_state = SensorStateManager()
        self.multi_processor = MultiProcessor(self.config, self.data_transformer, self.data_preprocessor, self.kafka_message_broker)
    
    def start_stream_processing(self) -> None:
        """
        Runs MTMC stream processing

        :return: None
        ::

            stream_processing_app.start_processing()
        """
        if self.config.io.enableDebug:
            make_clean_dir(self.config.io.outputDirPath)

        # Instantiate modules
        current_epoch_ts_ms = int(datetime.now().timestamp() * 1000)
        consumer_raw = self.kafka_message_broker.get_consumer(
            "mdx-raw", "mtmc-raw-consumer")
        consumer_notification = self.kafka_message_broker.get_consumer(
            "mdx-notification", f"mtmc-notification-consumer-{current_epoch_ts_ms}")
        producer = self.kafka_message_broker.get_producer()
        self.kafka_message_broker.get_consumed_notification_messages_and_count(consumer_notification)

        # Produce config request message
        logging.info(f"Requesting for the latest MTMC analytics config from mdx-notification")
        self.kafka_message_broker.produce_config_request(producer)

        # Initialize sensor state
        self.sensor_state.init_state(self.calibrator, self.calibration_path)
        sensor_state_objects = self.sensor_state.get_sensor_state_objects()
        self.data_preprocessor.set_sensor_state_objects(sensor_state_objects)

        # Initialize estimator of people's height
        self.people_height_estimator = PeopleHeightEstimator(self.config)
        self.data_preprocessor.set_people_height_estimator(self.people_height_estimator)

        del sensor_state_objects

        # Produce calibration request message
        logging.info(f"Requesting for the latest calibration from mdx-notification")
        self.kafka_message_broker.produce_calibration_request(producer)

        while True:
            self._batch_id += 1
            logging.info(f"Processing batch: {self._batch_id}")
            batch_start_time = datetime.utcnow()
            estimated_batch_end_time = batch_start_time + timedelta(seconds=self.config.streaming.kafkaMicroBatchIntervalSec)

            # Consume mdx-notification messages
            notification_messages, message_count = self.kafka_message_broker.get_consumed_notification_messages_and_count(consumer_notification)

            if message_count > 0:
                # Transform messages to notification objects
                map_type_to_notifications = self.data_transformer.transform_notification_messages(notification_messages, self.calibrator)
                del notification_messages
                logging.info(f"Transformed message(s) to notification object(s)...")

                if len(map_type_to_notifications["config"]) > 0:
                    logging.info(f"Received {len(map_type_to_notifications['config'])} configuration-related notifications.")
                    self.config_manager.process_notifications_and_update_config(map_type_to_notifications["config"], producer)

                if len(map_type_to_notifications["calibration"]) > 0:
                    logging.info(f"Received {len(map_type_to_notifications['calibration'])} calibration-related notifications.")

                    # Update sensor state objects
                    updated_sensor_ids = self.sensor_state.update_state(map_type_to_notifications["calibration"], self.calibrator)
                    del map_type_to_notifications
                    sensor_state_objects = self.sensor_state.get_sensor_state_objects()
                    logging.info(f"Updated {len(updated_sensor_ids)} sensor state object(s)...")

                    # Update places and locations of behavior state objects
                    self.behavior_state.update_places_and_locations(sensor_state_objects,
                                                                    updated_sensor_ids)
                    del updated_sensor_ids
                    logging.info(f"Updated place(s) and location(s) of behavior state object(s)...")

                    # Set sensor state objects for the preprocessor and Kafka module
                    self.data_preprocessor.set_sensor_state_objects(sensor_state_objects)
                    self.kafka_message_broker.set_sensor_state_objects(sensor_state_objects)
                    del sensor_state_objects
                    logging.info(f"Set sensor state object(s) for the preprocessor...")

            # Consume mdx-raw messages
            raw_messages, message_count = self.kafka_message_broker.get_consumed_raw_messages_and_count(consumer_raw)
            if message_count > 0:
                # Convert raw messages to behaviors
                batch_behaviors: List[Behavior] = list()
                if len(raw_messages) == 1:
                    batch_behaviors = self.data_transformer.transform_raw_messages_to_behaviors(raw_messages[0], self.data_preprocessor)
                elif len(raw_messages) > 1:
                    batch_behaviors = self.multi_processor.process_raw_message_lists(raw_messages)
                logging.info(f"Created a total of {len(batch_behaviors)} behavior(s) in batch: {self._batch_id}...")

                # Update people height
                self.data_preprocessor.update_people_height(batch_behaviors)

                # Update behavior state
                self.behavior_state.update_state(batch_behaviors, self.data_preprocessor)
                del batch_behaviors
                logging.info(f"Updated behavior state...")

                # Get all live behaviors from state
                live_behaviors = self.behavior_state.get_behaviors_in_state()
                logging.info(f"Got {len(live_behaviors)} live behavior(s) from state...")

                # Filter live behaviors
                live_behaviors = self.data_preprocessor.filter(live_behaviors)
                logging.info(f"Filtered live behavior(s)...")

                # Normalize embeddings of behaviors
                for i in range(len(live_behaviors)):
                    live_behaviors[i].embeddings = self.data_preprocessor.normalize_embeddings(live_behaviors[i].embeddings)
                logging.info(f"Normalized embedding(s) of live behavior(s)...")

                if self.config.io.enableDebug:
                    behaviors_json_path = os.path.join(self.config.io.outputDirPath, f"behaviors_{self._batch_id}.json")
                    write_pydantic_list_to_file(behaviors_json_path, live_behaviors)

                if len(live_behaviors) > 0:
                    # Cluster the behaviors to get MTMC objects
                    mtmc_objects, _, _ = self.clusterer.cluster(live_behaviors)
                    del live_behaviors
                    logging.info(f"Clustered live behavior(s) to get MTMC object(s)...")

                    # Update MTMC objects
                    updated_mtmc_objects = self.mtmc_state.get_updated_mtmc_objects(mtmc_objects, self._batch_id)
                    del mtmc_objects
                    logging.info(f"Updated MTMC object(s)...")

                    if self.config.io.enableDebug:
                        mtmc_objects_json_path = os.path.join(self.config.io.outputDirPath, f"mtmc_objects_{self._batch_id}.json")
                        write_pydantic_list_to_file(mtmc_objects_json_path, updated_mtmc_objects)

                    # Produce MTMC messages
                    self.kafka_message_broker.produce_mtmc_messages(producer, updated_mtmc_objects)
                    del updated_mtmc_objects

                # Delete older state
                self.behavior_state.delete_older_state()
                self.mtmc_state.delete_older_state(self.behavior_state.get_behavior_keys_in_state())
                logging.info(f"Processed batch: {self._batch_id}")

                batch_end_time = datetime.utcnow()
                if batch_end_time < estimated_batch_end_time:
                    time.sleep((estimated_batch_end_time - batch_end_time).total_seconds())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=validate_file_path, default="resources/app_mtmc_config.json",
                        action=ValidateFile, help="The input app config file")
    parser.add_argument("--calibration", type=validate_file_path, default="resources/calibration_building_k.json",
                        action=ValidateFile, help="The input calibration file")
    args = parser.parse_args()
    stream_processing_app = MTMCStreamProcessingApp(args.config, args.calibration)
    stream_processing_app.start_stream_processing()
