import pandas as pd
import logging
from config.logging_config import setup_logging
import requests
from dotenv import load_dotenv
import os
import json
import pika

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

connection = pika.BlockingConnection(
    pika.ConnectionParameters("localhost")
)
channel = connection.channel()

API_URL = os.environ.get("API_URL")

def predict_batch(df, epoch_id):
    logger.info("--- Predicting Batches ---")
    pdf = df.toPandas()

    if pdf.empty:
        return
    payload = pdf[
        [
            "Ambient_Temp_C",
            "Anode_Overhang_mm",
            "Electrolyte_Volume_ml",
            "Internal_Resistance_mOhm",
            "Capacity_mAh",
            "Retention_50Cycle_Pct",
            "Production_Line",
            "Shift",
            "Supplier",
        ]
    ].to_dict(orient="records")

    r = requests.post(API_URL, json={"dataframe_records": payload})

    logger.info(f"--- Status: {r.status_code} ---")
    logger.info(f"--- Response {r.json()} ---")


def publish_to_rabbit(df, epoch_id):

    pdf = df.toPandas()

    if pdf.empty:
        return

    payload = pdf[
    [
        "Ambient_Temp_C",
        "Anode_Overhang_mm",
        "Electrolyte_Volume_ml",
        "Internal_Resistance_mOhm",
        "Capacity_mAh",
        "Retention_50Cycle_Pct",
        "Production_Line",
        "Shift",
        "Supplier",
    ]].to_dict(orient="records")

    logger.info("--- Publishing to Rabbit ---")

    channel.basic_publish(
        exchange="",
        routing_key="prediction_requests_queue",
        body=json.dumps(payload)
    )
