from kafka import KafkaConsumer
from json import loads, dumps
import uuid

# Initialize the Kafka consumer
consumer = KafkaConsumer(
    'mdx-raw',
    bootstrap_servers='127.0.0.1:9092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id=str(uuid.uuid1()),
    value_deserializer=lambda x: loads(x.decode('utf-8'))
)

# Perform a dummy poll and seek to the end of the stream
consumer.poll()
consumer.seek_to_end()

# Open the file in append mode
with open('messages-cross.json', 'a') as json_file:
    # Start consuming messages
    for event in consumer:
        event_data = event.value
        # Write each message as a separate JSON object on a new line
        if('person' in event_data['objects'][0]):
         json_file.write(dumps(event_data) + '\n')

        # Optional break condition after consuming a certain number of messages
        # if some_condition: break
