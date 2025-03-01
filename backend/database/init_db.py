from database import Base, engine
from models import EnergyData

Base.metadata.create_all(engine)
