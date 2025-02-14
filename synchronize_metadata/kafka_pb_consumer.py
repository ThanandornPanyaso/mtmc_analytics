# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import google.protobuf.json_format as json_format
from kafka import KafkaConsumer
import sys
import json
import schema_pb2 as nvSchema
from typing import List, Dict
from kafka.consumer.fetcher import ConsumerRecord as KafkaConsumerRecord


class DataTransformer:
    """
    Module to transform data

    ::

        data_transformer = DataTransformer()
    """

    def _convert_bytes_to_protobuf_frame(self, message: bytes) -> nvSchema.Frame:
        """
        Converts message bytes to a protobuf frame object

        :param bytes message: message bytes
        :return: protobuf frame object
        :rtype: nvSchema.Frame
        """
        protobuf_frame = nvSchema.Frame().FromString(message)
        return protobuf_frame

    def transform_raw_messages_to_protobuf_frames(self, message: List[KafkaConsumerRecord]) -> List[nvSchema.Frame]:
        """
        Transforms raw messages to protobuf frame objects

        :param List[KafkaConsumerRecord] messages: list of Kafka consumer records
        :return: list of protobuf frame objects
        :rtype: List[nvSchema.Frame]
        ::

            protobuf_frames = data_transformer.transform_raw_messages_to_protobuf_frames(messages)
        """
        protobuf_frames: List[nvSchema.Frame] = list()
        protobuf_frames.append(self._convert_bytes_to_protobuf_frame(message.value))
        return protobuf_frames

f = open(sys.argv[1], "w")
data_transformer = DataTransformer()
consumer = KafkaConsumer(
            "test",
            bootstrap_servers="localhost:9092",
            group_id="raw-test-consumer",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            max_poll_interval_ms=900000,
            max_partition_fetch_bytes=2147483647
        )

while True:
    partitioned_messages = consumer.poll(timeout_ms=100,max_records=100000)
    for message in consumer:
        protobuf_frames = data_transformer.transform_raw_messages_to_protobuf_frames(message)
        for protobuf_frame in protobuf_frames:
            message_dict= json_format.MessageToDict(protobuf_frame,including_default_value_fields=True)
            f.write(json.dumps(message_dict))
            f.write("\n")
f.close()

