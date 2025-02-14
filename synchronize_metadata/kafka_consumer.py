# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import sys
from kafka import KafkaConsumer
from json import loads
import uuid 
import json


f = open(sys.argv[1], "w")

consumer = KafkaConsumer(
    'test',
    bootstrap_servers='127.0.0.1:9092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id=str(uuid.uuid1()),
    value_deserializer=lambda x: loads(x.decode('utf-8')),
)

# do a dummy poll to retrieve some message
consumer.poll()

# go to end of the stream
consumer.seek_to_end()

for i, event in enumerate(consumer):
    event_data = event.value
    f.write(json.dumps(event_data))
    f.write("\n")  # Add a newline character to separate each JSON string

f.close()

print(sys.argv[1])