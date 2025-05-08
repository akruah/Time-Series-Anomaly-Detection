import csv
import json
from time import sleep
from threading import Thread
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from kafka import KafkaProducer, KafkaConsumer
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
from slack_sdk import WebClient
import yagmail
import requests

app = FastAPI()

# Kafka Configuration
bootstrap_servers = 'localhost:9092'
topic_name = 'anomaly-events'

# Slack Configuration and Email (Replace with your Slack bot token)

def send_slack_message(anomaly: AnomalyEvent):
    webhook_url = "https://hooks.slack.com/services/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"  
    message = f"Equipment failure detected by sensor {anomaly.sensor_id} in zone {anomaly.zone} at {anomaly.timestamp}. Please investigate."
    payload = {"text": message}

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()  
        print(f"Slack message sent via webhook (sensor {anomaly.sensor_id})")
    except requests.exceptions.RequestException as e:
        print(f"Error sending Slack message via webhook: {e}")

# slack_token = "YOUR_SLACK_BOT_TOKEN"
# slack_channel = "#your-channel"
# slack_client = WebClient(token=slack_token)

sender_email = "akruah42@gmail.com"
sender_password = "M@ime42@Gaio"

# Event Schema using Pydantic
class AnomalyEvent(BaseModel):
    timestamp: str
    sensor_id: int
    zone: str
    predicted_label: int
    anomaly_type: str
    confidence: float

resolution_rules = {
    "temperature_spike": send_email_alert,
    "equipment_failure": send_slack_message,
    "inefficient_routing": send_email_alert,
    "high_humidity": send_slack_message,
    "inventory_mismatch": send_email_alert,
    "unauthorized_access": send_email_alert,
    "vibration_shock": send_slack_message
}
# Load Trained Model
model = LSTMAE(seq_len=60, n_feature=15, embeding_dim=128).to('cpu')
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
model.eval()
threshold = 0.041

# Inference Function
def infer_anomaly(sequence):
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        reconstruction = model(sequence_tonsor)
        loss = nn.SmoothL1loss(reduction='none')(reconstruction, sequence_tensor)
        score = loss.mean().item()
    return score

# Function to send email alerts
def send_email_alert(anomaly: AnomalyEvent):
    receiver_email = "st124418@ait.asia"  
    subject = f"Temperature Spike Alert: Sensor {anomaly.sensor_id} in Zone {anomaly.zone}"
    body = f"A temperature spike has been detected by sensor {anomaly.sensor_id} in zone {anomaly.zone} at {anomaly.timestamp}. Please investigate."

    try:
        yag = yagmail.SMTP(sender_email, sender_password)
        yag.send(to=receiver_email, subject=subject, contents=body)
        print(f"Email alert sent for temperature spike (sensor {anomaly.sensor_id})")
    except Exception as e:
        print(f"Error sending email alert: {e}")

# Function to send Slack messages
def send_slack_message(anomaly: AnomalyEvent):
    message = f"Equipment failure detected by sensor {anomaly.sensor_id} in zone {anomaly.zone} at {anomaly.timestamp}. Please investigate."

    try:
        response = slack_client.chat_postMessage(channel=slack_channel, text=message)
        if response["ok"]:
            print(f"Slack message sent for equipment failure (sensor {anomaly.sensor_id})")
        else:
            print(f"Error sending Slack message: {response['error']}")
    except Exception as e:
        print(f"Error sending Slack message: {e}")

# ========== Kafka Producer (CSV to Kafka) ==========
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda v: v.json().encode('utf-8')
)

def stream_csv_to_kafka():
    with open('anomalies.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row['timestamp'] = pd.to_datetime(row['timestamp']).isoformat() + "Z" 
            
            try:
                event = AnomalyEvent(**row)  
                producer.send(topic_name, value=event)
                print(f"Sending: {event.json()}")  
                sleep(2)  
            except ValueError as e:
                print(f"Error validating event: {e}, skipping row: {row}")

# ========== Kafka Consumer ==========
def kafka_event_listener():
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    for message in consumer:
        try:
            anomaly = AnomalyEvent(**message.value)
            resolve_anomaly(anomaly)
        except ValueError as e:
            print(f"Error validating anomaly event from Kafka: {e}, skipping message")

@app.on_event("startup")
def startup_event():
    Thread(target=kafka_event_listener, daemon=True).start()

# ========== Resolution Logic ==========
def resolve_anomaly(anomaly: AnomalyEvent):
    resolution_function = resolution_rules.get(anomaly.anomaly_type)
    if resolution_function:
        resolution_function(anomaly)
    else:
        print(f"Unknown anomaly type: {anomaly.anomaly_type}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
    
    
    
    
    
import csv
import json
from time import sleep
from threading import Thread
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from kafka import KafkaProducer, KafkaConsumer
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
from slack_sdk import WebClient
import yagmail
import requests
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from model import LSTMAE  # Make sure your LSTM-AE model class is correctly imported

app = FastAPI()

# Kafka Configuration
bootstrap_servers = 'localhost:9092'
topic_name = 'anomaly-events'

# Slack Configuration and Email
# Replace with your actual values
slack_token = "YOUR_SLACK_BOT_TOKEN"
slack_channel = "#your-channel"
slack_client = WebClient(token=slack_token)

sender_email = "akruah42@gmail.com"
sender_password = "M@ime42@Gaio"

# Event Schema using Pydantic
class AnomalyEvent(BaseModel):
    timestamp: str
    sensor_id: int
    zone: str
    predicted_label: int
    anomaly_type: str
    confidence: float

# Resolution Logic Functions
def send_email_alert(anomaly: AnomalyEvent):
    receiver_email = "st124418@ait.asia"
    subject = f"Temperature Spike Alert: Sensor {anomaly.sensor_id} in Zone {anomaly.zone}"
    body = f"A temperature spike has been detected by sensor {anomaly.sensor_id} in zone {anomaly.zone} at {anomaly.timestamp}. Please investigate."
    try:
        yag = yagmail.SMTP(sender_email, sender_password)
        yag.send(to=receiver_email, subject=subject, contents=body)
        print(f"Email alert sent (sensor {anomaly.sensor_id})")
    except Exception as e:
        print(f"Error sending email alert: {e}")

def send_slack_message(anomaly: AnomalyEvent):
    message = f"Anomaly ({anomaly.anomaly_type}) detected by sensor {anomaly.sensor_id} in zone {anomaly.zone} at {anomaly.timestamp}."
    try:
        response = slack_client.chat_postMessage(channel=slack_channel, text=message)
        if response["ok"]:
            print(f"Slack message sent (sensor {anomaly.sensor_id})")
        else:
            print(f"Slack error: {response['error']}")
    except Exception as e:
        print(f"Error sending Slack message: {e}")

resolution_rules = {
    "temperature_spike": send_email_alert,
    "equipment_failure": send_slack_message,
    "inefficient_routing": send_email_alert,
    "high_humidity": send_slack_message,
    "inventory_mismatch": send_email_alert,
    "unauthorized_access": send_email_alert,
    "vibration_shock": send_slack_message
}

# Load Trained Model
model = LSTMAE(seq_len=120, n_features=15, embedding_dim=128).to('cpu')
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
model.eval()
THRESHOLD = 0.027

# Inference Function
def infer_anomaly(sequence):
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        reconstruction = model(sequence_tensor)
        loss = nn.SmoothL1Loss(reduction='none')(reconstruction, sequence_tensor)
        score = loss.mean().item()
    return score

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda v: v.json().encode('utf-8')
)

# Real-time Inference + Stream to Kafka
def stream_csv_to_kafka():
    df = pd.read_csv('anomalies.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    features = df[['temperature', 'humidity', 'vibration', 'pressure', 'zone', 'sensor_id']].values
    timestamps = df['timestamp'].values

    seq_len = 120
    for i in range(len(df) - seq_len):
        seq = features[i:i+seq_len]
        score = infer_anomaly(seq)

        if score > THRESHOLD:
            row = df.iloc[i+seq_len-1]
            event = AnomalyEvent(
                timestamp=row['timestamp'].isoformat() + "Z",
                sensor_id=int(row['sensor_id']),
                zone=str(row['zone']),
                predicted_label=1,
                anomaly_type="equipment_failure",  # Replace with classifier if desired
                confidence=round(score, 5)
            )
            producer.send(topic_name, value=event)
            print(f"Anomalous sequence detected. Sending to Kafka: {event.json()}")

        sleep(1)

# Kafka Consumer
def kafka_event_listener():
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    for message in consumer:
        try:
            anomaly = AnomalyEvent(**message.value)
            resolve_anomaly(anomaly)
        except ValueError as e:
            print(f"Error validating anomaly from Kafka: {e}, skipping message")

@app.on_event("startup")
def startup_event():
    Thread(target=kafka_event_listener, daemon=True).start()
    Thread(target=stream_csv_to_kafka, daemon=True).start()

# Resolution Dispatcher
def resolve_anomaly(anomaly: AnomalyEvent):
    resolution_function = resolution_rules.get(anomaly.anomaly_type)
    if resolution_function:
        resolution_function(anomaly)
    else:
        print(f"Unknown anomaly type: {anomaly.anomaly_type}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
