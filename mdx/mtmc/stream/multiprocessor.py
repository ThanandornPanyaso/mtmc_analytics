# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import math
import datetime
import logging
import multiprocessing as mp
import threading
from queue import Queue as ThreadQueue
from typing import List, Dict, Any
from itertools import chain
from multiprocessing import shared_memory, Manager, Process

from mdx.mtmc.config import AppConfig
from mdx.mtmc.schema import Behavior
from mdx.mtmc.core.data import Preprocessor
from mdx.mtmc.stream.data_transformer import DataTransformer
from mdx.mtmc.stream.kafka_message_broker import KafkaMessageBroker


BYTE_COUNT_LIMIT = 9 ** 6


class MultiProcessor:
    """
    Module for multi-processing

    :param AppConfig config: configuration for the app
    :param DataTransformer data_transformer: data transformer
    :param Preprocessor data_preprocessor: data pre-processor
    :param KafkaMessageBroker kafka_message_broker: kafka message broker
    ::

        multi_processor = MultiProcessor(config, data_transformer, data_preprocessor, kafka_message_broker)
    """

    def __init__(self, config: AppConfig, data_transformer: DataTransformer, data_preprocessor: Preprocessor, kafka_message_broker: KafkaMessageBroker) -> None:
        self.config: AppConfig = config
        self.data_transformer: DataTransformer = data_transformer
        self.data_preprocessor: Preprocessor = data_preprocessor
        self.kafka_message_broker: KafkaMessageBroker = kafka_message_broker
 
    def _process_consumed_messages(self, process_idx: int, partitions_per_process: int, total_partitions: int, behavior_shared_list: List[Any]) -> None:
        """
        Processes consumed messages

        :param int process_idx: index of process
        :param int partitions_per_process: number of partitions per process
        :param int total_partitions: total number of partitions
        :param List[Any] behavior_shared_list: shared list of behaviors
        :return: None
        """
        try:
            partition_split_count_sharable_list = shared_memory.ShareableList(name="partition_split_count")
            current_idx = process_idx * partitions_per_process
            max_idx = current_idx + partitions_per_process - 1

            while (current_idx < total_partitions) and (current_idx <= max_idx):
                if partition_split_count_sharable_list[current_idx] == 1:
                    message_list = shared_memory.ShareableList(name=f"raw_data_{current_idx}")
                    behavior_shared_list[current_idx] = self.data_transformer.transform_raw_messages_to_behaviors(message_list, self.data_preprocessor)
                    message_list.shm.close()
                    message_list.shm.unlink()
                    del message_list
                else:
                    shared_lists: List[shared_memory.ShareableList] = list()
                    for i in range(partition_split_count_sharable_list[current_idx]):
                        shared_list = shared_memory.ShareableList(name=f"raw_data_{current_idx}_{i}")
                        shared_lists.append(shared_list)
                    message_list = list(chain.from_iterable(shared_lists))
                    behavior_shared_list[current_idx] = self.data_transformer.transform_raw_messages_to_behaviors(message_list, self.data_preprocessor)
                    for shared_list in shared_lists:
                        shared_list.shm.close()
                        shared_list.shm.unlink()
                    del shared_lists
                    del message_list
                current_idx += 1

            del partition_split_count_sharable_list
        except Exception as e:
            logging.error(f"Error in child: {e}")

    def _get_partition_split(self, message_lists: List[List[bytes]]) -> Dict[int, List[int]]:
        """
        Gets partition split information

        :param List[List[bytes]] message_lists: list of message lists
        :return: partition split information
        :rtype: Dict[int,List[int]]
        """
        split_info: Dict[int, List[int]] = dict()

        for i in range(len(message_lists)):
            byte_count = 0
            split_info[i]: List[int] = list()

            for j in range(len(message_lists[i])):
                if j == 0:
                    split_info[i].append(j)
                message = message_lists[i][j]
                byte_count += len(message)
                if byte_count >= BYTE_COUNT_LIMIT:
                    split_info[i].append(j)
                    byte_count = len(message)

        return split_info

    def process_raw_message_lists(self, message_lists: List[List[bytes]]) -> List[Behavior]:
        """
        Processes lists of raw messages

        :param List[List[bytes]] message_lists: list of message lists
        :return: list of behaviors
        :rtype: List[Behavior]
        ::

            behaviors = multi_processor.process_raw_message_lists(message_lists)
        """
        num_processes = mp.cpu_count()
        if num_processes > len(message_lists):
            num_processes = len(message_lists)

        manager = Manager()
        behavior_shared_list = manager.list()
        partitions_per_process = math.ceil(len(message_lists) / num_processes)
        partition_split_shm_restriction = self._get_partition_split(message_lists)
        partition_split_count = [len(split_start_index_list) for split_start_index_list in partition_split_shm_restriction.values()]
        partition_split_count_sharable_list = shared_memory.ShareableList(partition_split_count, name="partition_split_count")

        list_of_sharable_lists: List[shared_memory.ShareableList] = list()
        process_list: List[Process] = list()
        for i in range(len(message_lists)):
            message_list = message_lists[i]
            partition_split_start_indices = partition_split_shm_restriction[i]
            if len(partition_split_start_indices) > 1:
                for j in range(len(partition_split_start_indices)):
                    start_idx = partition_split_start_indices[j]
                    split_message_list = None
                    if j + 1 < len(partition_split_start_indices):
                        end_idx = partition_split_start_indices[j+1]
                        split_message_list = message_list[start_idx:end_idx]
                    else:
                        split_message_list = message_list[start_idx:]
                    shared_message_lists = shared_memory.ShareableList(split_message_list, name=f"raw_data_{i}_{j}")
                    list_of_sharable_lists.append(shared_message_lists)
            elif len(partition_split_start_indices) == 1:
                shared_message_lists = shared_memory.ShareableList(message_list, name=f"raw_data_{i}")
                list_of_sharable_lists.append(shared_message_lists)
            behavior_shared_list.append(None)
        del partition_split_shm_restriction

        for i in range(num_processes):
            process = Process(target=self._process_consumed_messages, args=(i, partitions_per_process, len(message_lists), behavior_shared_list))
            process_list.append(process)
            process.start()
            logging.info(f"Started child process: {i}")

        logging.info(f"Total number of processes: {len(process_list)}")

        counter=0
        for process in process_list:
            process.join(timeout=10)
            logging.info(f"Joined child process: {counter}")
            counter+=1

        del process_list

        for shared_list in list_of_sharable_lists:
            shared_list.shm.close()
        del list_of_sharable_lists

        partition_split_count_sharable_list.shm.close()
        partition_split_count_sharable_list.shm.unlink()
        del partition_split_count_sharable_list
        behavior_shared_list = list(filter(lambda x: x is not None, behavior_shared_list))
        behaviors = list(chain.from_iterable(behavior_shared_list))
        del manager
        del behavior_shared_list

        return behaviors

    def _get_partitioned_frame_lists_from_queue(self, queue: ThreadQueue) -> List[Any]:
        """
        Gets objects from a thread-safe queue

        :param ThreadQueue queue: queue containing objects
        :return: list of frames
        :rtype: List[Any]
        """
        queue_size = queue.qsize()
        frames: List[Any] = list()
        if queue_size > 0:
            num_consumed_items = 0
            while num_consumed_items < queue_size:
                frames.append(queue.get())
                num_consumed_items += 1
                queue.task_done()
        return frames

    def _get_timestamp_from_proto_ts(self, proto_ts: Any) -> datetime:
        """
        Gets timestamp from protobuf timestamp

        :param Any proto_ts: protobuf timestamp
        :return: timestamp
        :rtype: datetime
        """
        timestamp_ms = int((proto_ts.seconds + (proto_ts.nanos * (10 ** -9))) * 1000)
        timestamp_str = f"{datetime.datetime.utcfromtimestamp(timestamp_ms / 1000).isoformat(timespec='milliseconds')}Z"
        timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return timestamp

    def _get_behaviors_from_frames(self, idx_process: int, partitioned_protobuf_frame_queue: ThreadQueue, behavior_shared_list: List[List[Behavior]]) -> None:
        """
        Gets behaviors from frames

        :param int idx_process: index of process
        :param ThreadQueue partitioned_protobuf_frame_queue: a ThreadQueue containing partitioned frames
        :param List[List[Behavior]] behavior_shared_list: shared list of batched behaviors
        :return: None
        """
        exit_flag = False
        map_sensor_id_to_buffer_protobuf_frames: Dict[str, List[Any]] = dict()
        buffer_min_ts = None
        buffer_max_ts = None
        while True:
            partitioned_frame_lists = self._get_partitioned_frame_lists_from_queue(partitioned_protobuf_frame_queue)
            for partitioned_frame_list in partitioned_frame_lists:
                if partitioned_frame_list is None:
                    exit_flag = True
                    break

                map_sensor_id_to_batch_protobuf_frames: Dict[str, List[Any]] = dict()
                batch_min_ts = None
                batch_max_ts = None

                for _, frames in partitioned_frame_list:
                    for frame in frames:
                        sensor_id = frame.sensorId
                        if sensor_id not in map_sensor_id_to_batch_protobuf_frames:
                            map_sensor_id_to_batch_protobuf_frames[sensor_id] = list()
                        map_sensor_id_to_batch_protobuf_frames[sensor_id].append(frame)
                        frame_ts = self._get_timestamp_from_proto_ts(frame.timestamp)
                        if batch_min_ts is None:
                            batch_min_ts = frame_ts
                            batch_max_ts = frame_ts
                        elif frame_ts < batch_min_ts:
                            batch_min_ts = frame_ts
                        elif frame_ts > batch_max_ts:
                            batch_max_ts = frame_ts

                if buffer_max_ts is not None and batch_min_ts is not None and (batch_min_ts - buffer_max_ts).total_seconds() > self.config.streaming.mtmcPlusFrameBufferResetSec:
                    map_sensor_id_to_buffer_protobuf_frames.clear()
                    buffer_min_ts = None
                    buffer_max_ts = None

                if buffer_min_ts is None:
                    buffer_min_ts = batch_min_ts
                    buffer_max_ts = batch_max_ts
                    for sensor_id in map_sensor_id_to_batch_protobuf_frames.keys():
                        map_sensor_id_to_buffer_protobuf_frames[sensor_id] = map_sensor_id_to_batch_protobuf_frames[sensor_id]
                else:
                    if batch_min_ts is not None:
                        if batch_min_ts < buffer_min_ts:
                            buffer_min_ts = batch_min_ts

                        if batch_max_ts > buffer_max_ts:
                            buffer_max_ts = batch_max_ts

                        for sensor_id in map_sensor_id_to_batch_protobuf_frames.keys():
                            if sensor_id not in map_sensor_id_to_buffer_protobuf_frames:
                                map_sensor_id_to_buffer_protobuf_frames[sensor_id] = list()
                            map_sensor_id_to_buffer_protobuf_frames[sensor_id].extend(map_sensor_id_to_batch_protobuf_frames[sensor_id])

                if exit_flag or (buffer_max_ts is not None and buffer_min_ts is not None and (buffer_max_ts - buffer_min_ts).total_seconds() * 1000 >= self.config.streaming.mtmcPlusFrameBatchSizeMs):
                    batch_behavior_list = list()
                    sensors_in_buffer = list(map_sensor_id_to_buffer_protobuf_frames.keys())
                    for sensor_id in sensors_in_buffer:
                        frames = map_sensor_id_to_buffer_protobuf_frames[sensor_id]
                        behaviors = self.data_preprocessor.create_behaviors_from_protobuf_frames(frames)
                        batch_behavior_list.extend(behaviors)
                        logging.debug(f"[Process {idx_process}]: Obtained {len(behaviors)} behaviors from {len(frames)} frame messages of sensor ID: {sensor_id}.")

                    if len(batch_behavior_list) > 0:
                        behavior_shared_list.append(batch_behavior_list)

                    if len(sensors_in_buffer) > 0:
                        map_sensor_id_to_buffer_protobuf_frames.clear()
                        buffer_min_ts = None
                        buffer_max_ts = None

            if exit_flag:
                logging.info(f"[Process {idx_process}]: Exiting worker thread...")
                break

    def consume_raw_messages_and_add_to_behavior_list(self, idx_process: int, behavior_shared_list: List[Behavior]) -> None:
        """
        Consumes raw messages and adds to a behavior list

        :param int idx_process: index of process
        :param List[Behavior] behavior_shared_list: shared list of behaviors
        :return: None
        ::

            multi_processor.consume_raw_messages_and_add_to_behavior_list(idx_process, behavior_shared_list)
        """
        consumer_raw = self.kafka_message_broker.get_consumer("mdx-raw", "mtmc-plus-raw-consumer")
        partitioned_protobuf_frame_queue: ThreadQueue = ThreadQueue()
        worker_thread = threading.Thread(target=self._get_behaviors_from_frames, args=(idx_process, partitioned_protobuf_frame_queue, behavior_shared_list))
        worker_thread.start()

        while True:
            if mp.current_process().exitcode is not None:
                logging.info(f"[Process {idx_process}]: Child process interrupted. Stopping threads...")
                consumer_raw.close()
                partitioned_protobuf_frame_queue.put(None)
                worker_thread.join()
                partitioned_protobuf_frame_queue.join()          
                break

            partitioned_messages = self.kafka_message_broker.get_consumed_raw_messages(consumer_raw)
            
            partitioned_protobuf_messages = list()
            for partition, messages in partitioned_messages.items():
                logging.debug(f"[Process {idx_process}]: Processing partition ID {partition.partition} which has {len(messages)} messages")
                protobuf_frames = self.data_transformer.transform_raw_messages_to_protobuf_frames(messages)
                logging.debug(f"[Process {idx_process}]: Converted raw messages to protobuf frame objects")
                partitioned_protobuf_messages.append((partition.partition, protobuf_frames))
            partitioned_protobuf_frame_queue.put(partitioned_protobuf_messages)
