
import time,json
from kafka import KafkaProducer

# import pdb;pdb.set_trace()

broker = 'localhost:9092'
producer = KafkaProducer(bootstrap_servers=broker, value_serializer=lambda m: json.dumps(m).encode('utf-8'))
# producer = KafkaProducer(bootstrap_servers=broker)
data= open("/shareddata/data/activity-data/part-00007-tid-730451297822678341-1dda7027-2071-4d73-a0e2-7fb6a91e1d1f-0-c000.json","r").readlines()

print(">>> Now successfully start the kafka producer process. Run in background. Do not exit...")

for line in data:
    line = eval(line)
    time.sleep(1) 
    result = producer.send('lab09', line)

producer.flush()

# producer = KafkaProducer(retries=5)