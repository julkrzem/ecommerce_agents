from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests
from app.worker_process import process_msg

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    input: List[Message]

@app.post("/v1/chat")
async def process_endpoint(msg: ChatRequest):
    job = process_msg.delay(msg.model_dump())
    return {"task_id": job.id}



@app.get("/v1/results/{job_id}")
def get_result(job_id: str):

    from worker_process import app as celery_app
    result = celery_app.AsyncResult(job_id)
    if result.ready():
        return {"status": "done", "result": result.result}
    return {"status": "processing"}