from models import EnergyData

def process_sensor_data(data):
    # Business logic for handling incoming energy data
    energy_entry = EnergyData(**data)
    # Save to DB (pseudocode)
    save_to_db(energy_entry)
    return energy_entry
