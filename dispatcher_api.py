from fastapi import FASTAPI
from pydantic import BaseModel
import asyncio

app = FASTAPI()

class Anomaly(BaseModel):
   anomaly_type: str
   zone: int
   sensor_id: int
   confidence: float
   timestamp: str
   
@app.post("/resolve")
async def resolve_anomaly(anomaly: Anomaly):
   await asyncio.gather(
      send_mqtt(anomaly),
      send_email(anomaly),
      send_slack(anomaly)
   )
   return {"status": "dispatched"}

async def send_mqtt(anomaly):
   print(f"MQTT -> Zone {anomaly.zone} | Type: {anomaly.anomaly_type}")
   
async def send_email(anomaly):
   print(f"EMAIL -> Alert for {anomaly.anomaly_type} in zone {anomaly.zone}")
   
async def send_slack(anomaly):
   print(f"SLACK -> [{anomaly.sensor_id}] {anomaly.anomaly_type} (Confidence: {anomaly.confidence}")