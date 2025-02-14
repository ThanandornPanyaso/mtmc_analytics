# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
import json
from typing import List, Dict, Any
from kafka.consumer.fetcher import ConsumerRecord as KafkaConsumerRecord

from mdx.mtmc.stream.proto import schema_pb2 as nvSchema
from mdx.mtmc.schema import Behavior, Notification
from mdx.mtmc.core.data import Preprocessor
from mdx.mtmc.core.calibration import Calibrator


class DataTransformer:
    """
    Module to transform data

    ::

        data_transformer = DataTransformer()
    """

    def _convert_bytes_to_protobuf_frame(self, message: bytes) -> Any:
        """
        Converts message bytes to a protobuf frame object

        :param bytes message: message bytes
        :return: protobuf frame object
        :rtype: Any
        """
        protobuf_frame = nvSchema.Frame().FromString(message)
        return protobuf_frame

    def transform_raw_messages_to_protobuf_frames(self, messages: List[KafkaConsumerRecord]) -> List[Any]:
        """
        Transforms raw messages to protobuf frame objects

        :param List[KafkaConsumerRecord] messages: list of Kafka consumer records
        :return: list of protobuf frame objects
        :rtype: List[Any]
        ::

            protobuf_frames = data_transformer.transform_raw_messages_to_protobuf_frames(messages)
        """
        protobuf_frames: List[Any] = list()
        for message in messages:
            protobuf_frames.append(self._convert_bytes_to_protobuf_frame(message.value))
        return protobuf_frames

    def transform_raw_messages_to_behaviors(self, messages: List[bytes], data_preprocessor: Preprocessor) -> List[Behavior]:
        """
        Transforms raw messages to behavior objects

        :param List[bytes] messages: list of byte messages
        :param Preprocessor data_preprocessor: data preprocessor
        :return: list of behaviors
        :rtype: List[Behavior]
        ::

            behaviors = data_transformer.transform_raw_messages_to_behaviors(messages, data_preprocessor)
        """
        # Load raw messages to protobuf frames
        protobuf_frames: List[Any] = list()
        for message in messages:
            protobuf_frame = self._convert_bytes_to_protobuf_frame(message)
            protobuf_frames.append(protobuf_frame)

        # Create behaviors from protobuf frames
        behaviors = data_preprocessor.create_behaviors_from_protobuf_frames(protobuf_frames)

        return behaviors

    def transform_notification_messages(self, messages: List[KafkaConsumerRecord], calibrator: Calibrator) -> Dict[str, Notification]:
        """
        Transforms Kafka consumer records to notifications

        :param List messages: list of Kafka consumer records
        :param Calibrator calibrator: calibrator used to load sensors
        :return: map from type to notifications
        :rtype: Dict[str,Notification]
        ::

            map_type_to_notifications = data_transformer.transform_notification_messages(messages, calibrator)
        """
        map_type_to_notifications = {"calibration": list(), "config": list()}
        message_keys_to_be_processed = {"calibration", "mdx-mtmc-analytics-config"}
        for message in messages:
            message_key = message.key
            if (message_key is None) or (message_key.decode("utf-8") not in message_keys_to_be_processed):
                continue

            message_key = message_key.decode("utf-8")

            if message_key == "mdx-mtmc-analytics-config":
                event_type = None
                timestamp = None
                for header in message.headers:
                    if header[0] == "event.type":
                        event_type = header[1].decode("utf-8")
                        if event_type not in {"upsert-all", "upsert"}:
                            logging.error(f"ERROR: The event type {event_type} is not defined.")
                            exit(1)
                    elif header[0] == "timestamp":
                        timestamp = header[1].decode("utf-8")
                    else:
                        logging.error(f"ERROR: The header name {header[0]} is not defined.")
                        exit(1)
                if (event_type is None) or (timestamp is None):
                    logging.error(f"ERROR: Missing event type or timestamp in the notification message {message}.")
                    exit(1)

                map_type_to_notifications["config"].append(Notification(message=message.value.decode("utf-8"), event_type=event_type, timestamp=timestamp))
                del event_type
                del timestamp

            else:
                # Load list of sensors
                sensors = calibrator.load_sensors(json.loads(message.value.decode("utf-8")))

                # Load event type and timestamp
                event_type = None
                timestamp = None
                for header in message.headers:
                    if header[0] == "event.type":
                        event_type = header[1].decode("utf-8")
                        if event_type not in {"upsert-all", "upsert", "delete"}:
                            logging.error(f"ERROR: The event type {event_type} is not defined.")
                            exit(1)
                    elif header[0] == "timestamp":
                        timestamp = header[1].decode("utf-8")
                    else:
                        logging.error(f"ERROR: The header name {header[0]} is not defined.")
                        exit(1)
                if (event_type is None) or (timestamp is None):
                    logging.error(f"ERROR: Missing event type or timestamp in the notification message {message}.")
                    exit(1)

                map_type_to_notifications["calibration"].append(Notification(sensors=sensors, event_type=event_type, timestamp=timestamp))
                del event_type
                del timestamp

        for notification_type, notification_list in map_type_to_notifications.items():
            map_type_to_notifications[notification_type] = sorted(notification_list, key=lambda notification: notification.timestamp)

        return map_type_to_notifications
