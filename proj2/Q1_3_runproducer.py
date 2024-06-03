import time,json
import pandas as pd
from kafka import KafkaProducer

# import pdb;pdb.set_trace()

broker = 'localhost:64050'
producer = KafkaProducer(bootstrap_servers=broker, value_serializer=lambda m: json.dumps(m).encode('utf-8'))
# producer = KafkaProducer(bootstrap_servers=broker)

data = pd.read_csv("data/mydata/mytest.csv")
print(">>> Now successfully start the kafka producer process. Run in background. Do not exit...")

for idx, row in data.iterrows():
    message = {
        "input":row["input"],
        "output":row["output"]
    }
    producer.send('q1_3', message)
    # print(message)
    time.sleep(0.5)


producer.flush()

# producer = KafkaProducer(retries=5)
