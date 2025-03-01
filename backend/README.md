# Backend Service - Port Emissions and Energy Management System

## Overview
This is the backend service for the **Port Emissions and Energy Management System** project. It processes real-time data from IoT sensors, manages energy consumption data, and runs AI models to provide energy-saving recommendations. The backend also exposes a REST API to the frontend for data retrieval and interaction.

## Features
- **API for Real-Time Data**: Accepts energy consumption data from IoT sensors.
- **AI Integration**: Uses machine learning to forecast energy consumption and generate recommendations.
- **Database**: Stores energy usage data, emissions records, and system recommendations.
- **Background Tasks**: Celery is used for background processing (e.g., scheduling data aggregation, running AI models).

---

## Tech Stack
- **Python 3.9+**
- **FastAPI**: For building the RESTful API.
- **SQLAlchemy**: ORM for interacting with the database.
- **PostgreSQL**: Primary database for storing energy data.
- **Redis + Celery**: For task queue management.
- **TensorFlow**: For building and serving the machine learning model.
- **Docker**: For containerizing the backend service.

---

## Folder Structure
```
/backend
│
├── api/
│   ├── app.py               # Main entry point for the FastAPI server
│   ├── config.py            # Configuration file for environment variables
│   ├── models.py            # SQLAlchemy database models
│   ├── routes.py            # API routes (sensor data, recommendations)
│   ├── tasks.py             # Background task definitions using Celery
│   └── utils.py             # Utility functions (logging, validation)
│
├── ai_model/
│   ├── model.py             # AI model for energy forecasting
│   ├── train.py             # Training script for the AI model
│   ├── preprocess.py        # Preprocessing steps for historical data
│   ├── predict.py           # Inference script for running predictions
│   └── saved_models/        # Directory to store trained models
│
├── database/
│   ├── init_db.py           # Script to initialize the database schema
│   └── schema.sql           # SQL script for the database schema (if needed)
│
├── services/
│   ├── energy_service.py    # Service logic for energy data processing
│   └── recommendation_service.py  # Logic for generating recommendations
│
├── tests/
│   ├── test_api.py          # Unit tests for API routes
│   └── test_models.py       # Unit tests for database models
│
├── requirements.txt         # Python dependencies
├── Dockerfile               # Dockerfile for containerizing the backend
└── README.md                # Documentation for the backend service
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/port-energy-management-system.git
cd port-energy-management-system/backend
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the `backend/` directory with the following variables:
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/port_energy
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your_secret_key
```

### 5. Initialize the Database
Ensure that PostgreSQL is running and initialize the database schema:
```bash
python database/init_db.py
```

---

## Running the Backend Service

### 1. Start Redis (For Celery Tasks)
Make sure Redis is running (you can use Docker):
```bash
docker run -p 6379:6379 -d redis:alpine
```

### 2. Run Celery Worker
Start the Celery worker for background task processing:
```bash
celery -A api.tasks worker --loglevel=info
```

### 3. Start the FastAPI Server
Run the backend server:
```bash
uvicorn api.app:app --reload
```
The backend will be available at `http://localhost:5000`.

---

## Running Tests

To run the unit tests for the backend API and models:
```bash
pytest tests/
```

---

## Dockerization

The backend service is Dockerized for easy deployment. To build and run the backend service using Docker:

### 1. Build the Docker Image
```bash
docker build -t port-energy-backend .
```

### 2. Run the Docker Container
```bash
docker run -p 5000:5000 --env-file .env port-energy-backend
```

---

## Key API Endpoints

- **POST /upload-sensor-data/**  
  Accepts energy consumption data from IoT sensors.
  ```json
  {
    "timestamp": "2024-01-01T12:00:00Z",
    "machine_id": "crane1",
    "energy_consumption": 200.5
  }
  ```

- **GET /recommendations/**  
  Returns AI-generated energy-saving recommendations.
  ```json
  [
    "Shift crane operations to off-peak hours.",
    "Reduce lighting during non-operational hours."
  ]
  ```

---

## AI Model Workflow

The backend includes an AI model for energy forecasting. Here’s how the model is trained and used:
1. **Training**: Run the `train.py` script in the `ai_model/` folder to train the model using historical energy consumption data.
   ```bash
   python ai_model/train.py
   ```
2. **Inference**: Use the trained model to make predictions by running the `predict.py` script.
   ```bash
   python ai_model/predict.py
   ```












