# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
import json
from typing import List, Dict, Tuple, Any
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.consumer.fetcher import ConsumerRecord as KafkaConsumerRecord

from mdx.mtmc.schema import convert_datetime_to_iso_8601_with_z_suffix, MTMCObject, MTMCObjectsPlus, SensorStateObject
from mdx.mtmc.config import AppConfig


def json_serializer(object_instance):
    if isinstance(object_instance, datetime):
        return convert_datetime_to_iso_8601_with_z_suffix(object_instance)
    raise TypeError("Type not serializable")


class KafkaMessageBroker:
    """
    Module for Kafka message broker

    :param AppConfig config: configuration for the app
    ::

        kafka_message_broker = KafkaMessageBroker(config)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self._sensor_state_objects: Dict[str, SensorStateObject] = dict()

    def set_sensor_state_objects(self, sensor_state_objects: Dict[str, SensorStateObject]) -> None:
        """
        Sets sensor state objects

        :param Dict[str,SensorStateObject] sensor_state_objects: map from sensor IDs to sensor state objects
        :return: None
        ::

            kafka_message_broker.set_sensor_state_objects(sensor_state_objects)
        """
        self._sensor_state_objects = sensor_state_objects

    def get_consumer(self, topic: str, group_id: str) -> KafkaConsumer:
        """
        Creates Kafka consumer

        :param str topic: Kafka topic
        :param str group_id: group ID
        :return: Kafka consumer
        :rtype: KafkaConsumer
        ::

            consumer = kafka_message_broker.get_consumer(topic, group_id)
        """
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.config.streaming.kafkaBootstrapServers,
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            max_poll_interval_ms=900000,
            max_partition_fetch_bytes=2147483647
        )
        return consumer

    def get_producer(self) -> KafkaProducer:
        """
        Creates Kafka producer

        :return: Kafka producer
        :rtype: KafkaProducer
        ::

            producer = kafka_message_broker.get_producer()
        """
        producer = KafkaProducer(
            linger_ms=self.config.streaming.kafkaProducerLingerMs,
            bootstrap_servers=self.config.streaming.kafkaBootstrapServers,
            key_serializer=lambda k: k.encode("utf-8"),
            value_serializer=lambda v: v.encode("utf-8")
        )
        return producer

    def get_consumed_raw_messages_and_count(self, consumer: KafkaConsumer) -> Tuple[List[Any], int]:
        """
        Consumes messages and count from the mdx-raw topic

        :param KafkaConsumer consumer: Kafka consumer
        :rtype: Tuple[List[Any],int]
        ::

            messages, message_count = kafka_message_broker.get_consumed_raw_messages_and_count(consumer)
        """
        partitioned_messages = self.get_consumed_raw_messages(consumer)
        message_count = 0
        for partition, messages in partitioned_messages.items():
            message_count += len(messages)
            message_values = [message.value for message in messages]
            partitioned_messages[partition] = message_values
        logging.info(f"Consumed {message_count} raw message(s)")
        return list(partitioned_messages.values()), message_count

    def get_consumed_raw_messages(self, consumer: KafkaConsumer) -> Dict[TopicPartition, List[KafkaConsumerRecord]]:
        """
        Consumes partitioned messages from the mdx-raw topic

        :param KafkaConsumer consumer: Kafka consumer
        :rtype: Dict[TopicPartition,List[KafkaConsumerRecord]]
        ::

            partitioned_messages = kafka_message_broker.get_consumed_raw_messages(consumer)
        """
        partitioned_messages = consumer.poll(
            timeout_ms=self.config.streaming.kafkaRawConsumerPollTimeoutMs,
            max_records=self.config.streaming.kafkaConsumerMaxRecordsPerPoll
        )
        return partitioned_messages

    def get_consumed_notification_messages_and_count(self, consumer: KafkaConsumer) -> Tuple[List[KafkaConsumerRecord], int]:
        """
        Consumes messages and count from the mdx-notification topic

        :param KafkaConsumer consumer: Kafka consumer
        :return: Kafka consumer records and the count of messages
        :rtype: Tuple[List[KafkaConsumerRecord],int]
        ::

            messages, message_count = kafka_message_broker.get_consumed_notification_messages_and_count(consumer)
        """
        messages: List[KafkaConsumerRecord] = list()
        partitioned_messages = consumer.poll(
            timeout_ms=self.config.streaming.kafkaNotificationConsumerPollTimeoutMs,
            max_records=self.config.streaming.kafkaConsumerMaxRecordsPerPoll
        )
        for partition in partitioned_messages.keys():
            messages += partitioned_messages[partition]
        del partitioned_messages
        message_count = len(messages)
        logging.info(f"Consumed {message_count} notification message(s)")
        return messages, message_count

    def produce_mtmc_messages(self, producer: KafkaProducer, mtmc_objects: List[MTMCObject]) -> None:
        """
        Sends messages containing MTMC objects to mdx-mtmc

        :param KafkaProducer producer: Kafka producer
        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :return: None
        ::

            kafka_message_broker.produce_mtmc_messages(producer, mtmc_objects)
        """
        for mtmc_object in mtmc_objects:
            mtmc_object_dict = mtmc_object.dict(exclude_none=True)
            del mtmc_object

            for i in range(len(mtmc_object_dict["matched"])):
                mtmc_object_dict["matched"][i].pop("key", None)

                # Remove startFrame and endFrame before sending via Kafka only if the sensor's source is not nvstreamer
                sensor_state_object = self._sensor_state_objects.get(mtmc_object_dict["matched"][i]["sensorId"], None)
                if sensor_state_object is not None:
                    attributes = sensor_state_object.sensor.attributes
                    source = None
                    for attribute in attributes:
                        if attribute["name"] == "source":
                            source = attribute["value"]
                            break
                    if source == "nvstreamer":
                        start_frame = int(mtmc_object_dict["matched"][i]["startFrame"])
                        end_frame = int(mtmc_object_dict["matched"][i]["endFrame"])
                        if start_frame > end_frame:
                            logging.warning(f"WARNING: The source for behavior ID {mtmc_object_dict['matched'][i]['id']} is nvstreamer. "
                                            f"but the starting frame {start_frame} is larger than the ending frame {end_frame}.")
                        continue
                mtmc_object_dict["matched"][i].pop("startFrame", None)
                mtmc_object_dict["matched"][i].pop("endFrame", None)

            producer.send("mdx-mtmc", json.dumps(mtmc_object_dict, default=json_serializer), key=mtmc_object_dict["globalId"])
            del mtmc_object_dict

        producer.flush()
        logging.info(f"Produced {len(mtmc_objects)} message(s) of MTMC object(s)")

    def produce_mtmc_plus_messages(self, producer: KafkaProducer, mtmc_objects_plus: MTMCObjectsPlus) -> None:
        """
        Sends messages containing MTMC objects plus locations to mdx-rtls

        :param KafkaProducer producer: Kafka producer
        :param MTMCObjectsPlus mtmc_objects_plus: MTMC objects plus locations
        :return: None
        ::

            kafka_message_broker.produce_mtmc_plus_messages(producer, mtmc_objects_plus)
        """
        mtmc_objects_plus_dict = mtmc_objects_plus.dict(exclude_none=True)

        if not self.config.io.enableDebug:
            mtmc_objects_plus_dict.pop("frameId", None)

        locations_of_objects: List[Dict[str, Any]] = list()
        for i in range(len(mtmc_objects_plus_dict["locationsOfObjects"])):
            mtmc_objects_plus_dict["locationsOfObjects"][i].pop("matchedBehaviorKeys", None)
            if self.config.clustering.enableOnlineDynamicUpdate and (int(mtmc_objects_plus_dict["locationsOfObjects"][i]["id"]) >= 0):
                locations_of_objects.append(mtmc_objects_plus_dict["locationsOfObjects"][i])
        if self.config.clustering.enableOnlineDynamicUpdate:
            mtmc_objects_plus_dict["locationsOfObjects"] = locations_of_objects

        if (not self.config.streaming.sendEmptyMtmcPlusMessages) and (len(mtmc_objects_plus_dict["locationsOfObjects"]) == 0):
            return

        producer.send("mdx-rtls", json.dumps(mtmc_objects_plus_dict, default=json_serializer), key=mtmc_objects_plus_dict["place"])
        producer.flush()
        logging.info(f"Produced a message of {len(mtmc_objects_plus_dict['locationsOfObjects'])} MTMC object(s) plus location(s) "
                     f"at timestamp {mtmc_objects_plus.timestamp} and frame ID {mtmc_objects_plus.frameId}")
        del mtmc_objects_plus
        del mtmc_objects_plus_dict

    def produce_calibration_request(self, producer: KafkaProducer) -> None:
        """
        Sends a message requesting calibration to mdx-notification

        :param KafkaProducer producer: Kafka producer
        :return: None
        ::

            kafka_message_broker.produce_calibration_request(producer)
        """
        producer.send("mdx-notification", key="request-calibration", value="")
        producer.flush()
        logging.info(f"Produced a message of calibration request")

    def produce_config_request(self, producer: KafkaProducer) -> None:
        """
        Sends a message requesting MTMC analytics config

        :param KafkaProducer producer: Kafka producer
        :return: None
        ::

            kafka_message_broker.produce_config_request(producer)
        """
        producer.send("mdx-notification", key="request-mdx-mtmc-analytics-config", value="")
        producer.flush()
        logging.info(f"Produced a message of config request")
    
    def produce_init_config(self, producer: KafkaProducer) -> None:
        """
        Sends a message containing init MTMC analytics config

        :param KafkaProducer producer: Kafka producer
        :return: None
        ::

            kafka_message_broker.produce_init_config(producer)
        """
        non_updatable_categories = {"io", "streaming"}
        config_to_be_sent = self.config.dict()
        for category in non_updatable_categories:
            del config_to_be_sent[category]
        producer.send("mdx-notification", key="init-mdx-mtmc-analytics-config", value=json.dumps(config_to_be_sent))
        logging.info(f"Produced a message containing init config.")
