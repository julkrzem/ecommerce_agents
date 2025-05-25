from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import json 

from app.agents.conversation_model import Chat

app = FastAPI()
chat = Chat()

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
async def chat_endpoint(request: ChatRequest):

    if len(request.input)>=5:
        history = [m.content for m in request.input[-5:-1]]
    else:
        history = [m.content for m in request.input[:-1]]

    user_input = request.input[-1].content
    response = json.dumps(chat.run(user_input, history))

    return response