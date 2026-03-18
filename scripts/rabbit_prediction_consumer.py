import pika
import json
import pandas as pd
import joblib
import os
import requests

API_URL = os.environ.get("API_URL")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")

def callback(ch, method, properties, body):
    samples = json.loads(body)

    r = requests.post(
        API_URL,
        json={"dataframe_records": samples}
    )

    if r.status_code != 200:
        print("MLflow error:", r.text)
        ch.basic_nack(delivery_tag=method.delivery_tag)
        return

    preds = r.json()["predictions"]
    
    for sample, pred in zip(samples, preds):
        result = {
            "request": sample,
            "prediction": pred
        }

        ch.basic_publish(
            exchange="",
            routing_key="prediction_result_queue",
            body=json.dumps(result)
        )

connection = pika.BlockingConnection(
    pika.ConnectionParameters(RABBITMQ_HOST)
)

channel = connection.channel()
#channel.queue_declare(queue="prediction_queue")

channel.basic_consume(
    queue="prediction_requests_queue",
    on_message_callback=callback,
    auto_ack=True
)

print("Waiting for prediction requests...")
channel.start_consuming()


