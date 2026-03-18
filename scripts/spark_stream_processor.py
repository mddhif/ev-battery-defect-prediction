import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType
from pyspark.sql.functions import from_json, col, avg, window
import requests
import os
from dotenv import load_dotenv
from config.logging_config import setup_logging
import logging
import pika
from spark_stream_helpers import predict_batch, publish_to_rabbit

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

connection = pika.BlockingConnection(
    pika.ConnectionParameters("localhost")
)
channel = connection.channel()

API_URL = os.environ.get("API_URL")

spark = SparkSession.builder \
    .appName("EVBatteryDataStream") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


schema = (
    StructType()
    .add("timestamp", StringType(), True)
    .add("Cell_ID", StringType(), True)
    .add("Batch_ID", StringType(), True)
    .add("Production_Line", StringType(), True)
    .add("Shift", StringType(), True)
    .add("Supplier", StringType(), True)
    .add("Ambient_Temp_C", DoubleType(), True)
    .add("Anode_Overhang_mm", DoubleType(), True)
    .add("Electrolyte_Volume_ml", DoubleType(), True)
    .add("Internal_Resistance_mOhm", DoubleType(), True)
    .add("Capacity_mAh", DoubleType(), True)
    .add("Retention_50Cycle_Pct", DoubleType(), True)
)

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "incoming_data") \
    .option("startingOffsets", "earliest") \
    .load()

parsed = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*") \
    .filter(col("Capacity_mAh") > 0)

agg = parsed.groupBy(
    window(col("timestamp"), "1 minute"),
    col("Production_Line")
).agg(avg("Capacity_mAh").alias("avg_capacity"))

rabbit_query = parsed.writeStream \
    .foreachBatch(publish_to_rabbit) \
    .trigger(processingTime="5 seconds") \
    .start()

monitoring = agg.writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

'''query = parsed.writeStream \
    .format("console") \
    .start()'''

#rabbit_query.awaitTermination()
spark.streams.awaitAnyTermination()