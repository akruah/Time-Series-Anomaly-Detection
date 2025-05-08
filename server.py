from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging

app = FastAPI()

logging.basicConfig(filename="resolved_anomalies.log", level=logging.INFO, format="%(asctime)s - %(message)s")

class AnomalyEvent(BaseModel):
    timestamp: str
    sensor_id: int
    zone: int
    predicted_label: int
    anomaly_type: str
    confidence: float

@app.post("/resolve")
async def resolve_anomaly(event: AnomalyEvent):
    # Log the resolved anomaly
    log_message = (
        f"Resolved anomaly from sensor {event.sensor_id} in zone {event.zone} - "
        f"Type: {event.anomaly_type}, Confidence: {event.confidence}"
    )
    logging.info(log_message)
    return {"status": "success", "message": log_message}
