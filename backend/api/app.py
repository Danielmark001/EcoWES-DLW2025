from fastapi import FastAPI
from routes import energy_routes

app = FastAPI()

app.include_router(energy_routes)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
