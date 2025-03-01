from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base

class EnergyData(Base):
    __tablename__ = 'energy_data'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    machine_id = Column(String)
    energy_consumption = Column(Float)
