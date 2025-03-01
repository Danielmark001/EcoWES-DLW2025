from celery import Celery
from services import recommendation_service

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def process_energy_data():
    recommendation_service.generate_recommendations()
