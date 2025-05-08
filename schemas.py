from pydantic import BaseModel

class AnomalyEvent(BaseModel):
   timestamp: str
   sensor_id: int
   zone: int
   predicted_label: int
   anomaly_type: str
   confidence: float
   