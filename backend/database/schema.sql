CREATE TABLE energy_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ,
    machine_id VARCHAR(255),
    energy_consumption FLOAT
);
