from kafka import KafkaConsumer
import json 

broker = 'localhost:9092'
consumer = KafkaConsumer('lab09', 
        bootstrap_servers=broker, 
        value_deserializer=lambda m: json.loads(m.decode('utf-8')), 
        consumer_timeout_ms=100000,
        auto_offset_reset='earliest', enable_auto_commit=False
    )

for message in consumer:
    print(f"The value of offset {message.offset} is >>> {message.value}.")

