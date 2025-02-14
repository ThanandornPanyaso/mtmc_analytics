from kafka import KafkaConsumer
from json import loads
import uuid

consumer = KafkaConsumer(
    'mdx-raw',
    bootstrap_servers='127.0.0.1:9092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id=str(uuid.uuid1()),
    value_deserializer=lambda x: loads(x.decode('utf-8'))
)

# do a dummy poll to retrieve some message
consumer.poll()

# go to end of the stream
consumer.seek_to_end()

for i, event in enumerate(consumer):
    print(event)
       
        