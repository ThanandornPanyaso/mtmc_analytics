# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
import json
from typing import List
from kafka import KafkaProducer
from mdx.mtmc.config import AppConfig
from mdx.mtmc.schema import Notification
from mdx.mtmc.stream.kafka_message_broker import KafkaMessageBroker


class ConfigManager:
    """
    Module to manage configuration

    :param AppConfig config: configuration for the app
    :param KafkaMessageBroker kafka_message_broker: Kafka message broker
    ::

        config_manager = ConfigManager(config, kafka_message_broker)
    """

    def __init__(self, config: AppConfig, kafka_message_broker: KafkaMessageBroker) -> None:
        self.config: AppConfig = config
        self.kafka_message_broker: KafkaMessageBroker = kafka_message_broker

    def _update_config(self, updated_config_str: str) -> None:
        """
        Updates configuration

        :param str updated_config_str: Updated configuration string
        :return: None
        """
        updated_config = json.loads(updated_config_str)
        non_updatable_categories = {"io", "streaming"}
        for category in updated_config.keys():
            if category not in self.config.__dict__.keys():
                logging.warning(f"WARNING: Ignoring the invalid config category: {category}.")
                continue

            if category in non_updatable_categories:
                logging.warning(f"WARNING: Ignoring config category: {category} as it is not updatable.")
                continue

            config_category = self.config.__getattribute__(category)
            for config_name in updated_config[category]:
                if config_name not in config_category.__dict__.keys():
                    logging.warning(f"WARNING: Ignoring the invalid config: {category}.{config_name}.")
                    continue
                config_category.__setattr__(config_name,updated_config[category][config_name])

        logging.info("Completed updating the app config.")

    def process_notifications_and_update_config(self, config_notifications: List[Notification], producer: KafkaProducer):
        """
        Processes notifications and updates configuration

        :param List[Notification] config_notifications: notifications of configuration
        :param KafkaProducer producer: Kafka producer
        :return: None
        ::

            config_manager.process_notifications_and_update_config(config_notifications, producer)
        """
        for notification in config_notifications:
            if notification.message == "":
                if notification.event_type == "upsert-all":
                    self.kafka_message_broker.produce_init_config(producer)
                else:
                    logging.warning(f"WARNING: Ignoring the empty config for event type: {notification.event_type}.")
                    continue
            else:
                self._update_config(notification.message)
        logging.info("Updated config based on all the notification messages...")
