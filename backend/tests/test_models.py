from api.models import EnergyData
from datetime import datetime

def test_energy_data_model():
    # Create a sample EnergyData object
    energy_data = EnergyData(
        timestamp=datetime.now(),
        machine_id="crane1",
        energy_consumption=150.0
    )

    # Test that the data was initialized correctly
    assert energy_data.machine_id == "crane1"
    assert energy_data.energy_consumption == 150.0
