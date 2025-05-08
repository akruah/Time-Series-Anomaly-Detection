import csv
import json
from time import sleep
from kafka import KafkaProducer
from schemas import AnomalyEvent
import pandas as pd


bootstrap_servers = 'localhost:9092'
topic_name = 'anomalies'

#Kafka Producer Setup
producer = KafkaProducer(
   bootstrap_servers=bootstrap_servers,
   value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def stream_csv_to_kafka(file_path = r'C:\anomalies.csv', delay=2):
   df = pd.read_csv(file_path)
   
   
   for _, row in df.iterrows():
        try:
            event = AnomalyEvent(
                timestamp=pd.to_datetime(row['timestamp']).isoformat(),
                sensor_id=int(row['sensor_id']),
                zone=int(row['zone']),
                predicted_label=int(row['predicted_label']),
                anomaly_type=row['anomaly_type'],
                confidence=float(row['confidence'])
            )
            producer.send(topic_name, value=event.dict())
            print(f"Sent to Kafka: {event}")
            sleep(delay)
        except Exception as e:
            print(f"Error processing row {row}: {e}")

if __name__ == "__main__":
    stream_csv_to_kafka() 
