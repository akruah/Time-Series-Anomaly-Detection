import json
from kafka import KafkaConsumer
from schemas import AnomalyEvent
from resolvers.email_alert import send_email_alert
from resolvers.slack_alert import send_slack_message
from resolvers.mqtt_alert import send_mqtt_alert

bootstrap_servers = 'localhost:9092'
topic_name = 'anomalies'  

resolvers_rules = {
    "temperature_spike": send_email_alert,
    "equipment_failure": send_slack_message,
    "inefficient_routing": send_email_alert,
    "high_humidity": send_slack_message,
    "inventory_mismatch": send_email_alert,
    "unauthorized_access": send_email_alert,
    "vibration_shock": send_slack_message
}

def resolve_anomaly(anomaly: AnomalyEvent):
    resolver = resolvers_rules.get(anomaly.anomaly_type)
    if resolver:
        print(f"Resolving anomaly: {anomaly.anomaly_type} (Sensor {anomaly.sensor_id})")
        resolver(anomaly)
        send_mqtt_alert(anomaly)
    else:
        print(f"No resolver defined for anomaly type: {anomaly.anomaly_type}. Skipping...")

def kafka_event_listener():
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=[bootstrap_servers],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='anomaly-resolvers'
    )
    print(f"Listening to kafka topic '{topic_name}' for anomalies...")

    for msg in consumer:
        try:
            event_data = msg.value
            anomaly = AnomalyEvent(**event_data)
            if anomaly.anomaly_type != "No Anomaly":
                resolve_anomaly(anomaly)
            else:
                print(f"No anomaly detected. Sensor {anomaly.sensor_id}. Skipping...")
        except Exception as e:
            print(f"Error processing message: {e}")

if __name__ == "__main__":
    kafka_event_listener()
