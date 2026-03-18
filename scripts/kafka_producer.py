import pandas as pd
from kafka import KafkaProducer
import json
import time


producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8") 
)

topic = "incoming_data"
df = pd.read_csv("data/producer_test_data.csv")

for _, row in df.iterrows():
    producer.send(topic, row.to_dict())
    print("Raw data sent :", row.to_dict())
    time.sleep(4)

producer.flush()
