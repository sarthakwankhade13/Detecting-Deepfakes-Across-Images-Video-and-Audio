from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.detect import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def home():
    return {"message": "Backend is working 🚀"}
