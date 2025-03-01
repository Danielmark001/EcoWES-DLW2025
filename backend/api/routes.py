from fastapi import APIRouter
from models import EnergyData
from services import energy_service

energy_routes = APIRouter()

@energy_routes.post("/upload-sensor-data")
async def upload_sensor_data(data: EnergyData):
    result = energy_service.process_sensor_data(data)
    return {"status": "success", "result": result}
