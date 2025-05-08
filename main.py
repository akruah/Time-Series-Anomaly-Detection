import streamlit as st 
from kafka import KafkaConsumer
import threading
import json
import time
import pandas as pd
import os
import requests
import plotly.express as px

TOPIC_NAME = os.getenv('TOPIC_NAME', 'anomalies')  
KAFKA_SERVER = os.getenv('KAFKA_SERVER', 'localhost:9092')

st.set_page_config(page_title="Anomaly Resolution Dashboard", layout="wide")
st.title("Real-Time Anomaly Resolution Dashboard")

if 'anomalies' not in st.session_state:
    st.session_state.anomalies = []

@st.cache_resource
def connect_consumer():
    try:
        consumer = KafkaConsumer(
            TOPIC_NAME,
            bootstrap_servers=[KAFKA_SERVER],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='streamlit-dashboard'
        )
        return consumer
    except Exception as e:
        st.error(f"Error connecting to Kafka broker: {e}")
        return None

consumer = connect_consumer()

# Layout
col1, col2 = st.columns(2)
placeholder = st.empty()

def consume_messages():
    if consumer is None:
        st.error("Kafka Consumer not available. Please check Kafka server and topic.")
        return

    for message in consumer:
        anomaly_data = message.value
        st.session_state.anomalies.append(anomaly_data)

        # Dispatch anomaly to FastAPI Resolution Endpoint
        try:
            requests.post("http://localhost:8000/resolve", json=anomaly_data)
        except Exception as e:
            st.warning(f"Failed to dispatch resolution: {e}")

        df = pd.DataFrame(st.session_state.anomalies)

        with placeholder.container():
            st.dataframe(df, use_container_width=True, height=500)

        with col1:
            st.metric("Total Anomalies", len(df))

        with col2:
            if 'anomaly_type' in df.columns:
                types = df['anomaly_type'].value_counts()
                st.subheader("Anomaly Types")
                for anomaly_type, count in types.items():
                    st.write(f"**{anomaly_type}** : {count}")

        # Visualizations
        if not df.empty and 'anomaly_type' in df.columns:
            st.markdown("##Anomaly Visualizations")

            tab1, tab2, tab3 = st.tabs(["📎 Type Distribution", "⏱ Time Series", "Zone Distribution"])

            with tab1:
                fig1 = px.pie(df, names='anomaly_type', title='Anomaly Type Distribution')
                st.plotly_chart(fig1, use_container_width=True)

            with tab2:
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    df['minute'] = df['time'].dt.floor('min')
                    time_counts = df.groupby('minute').size().reset_index(name='count')
                    fig2 = px.line(time_counts, x='minute', y='count', title='Anomaly Frequency Over Time')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No 'time' column available for time series plot.")

            with tab3:
                if 'zone' in df.columns:
                    fig3 = px.bar(df, x='zone', color='anomaly_type', title='Zone-wise Anomaly Count')
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No 'zone' column available for zone-wise plot.")

        time.sleep(1)

consume_messages()
