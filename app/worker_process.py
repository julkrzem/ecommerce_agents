from pydantic import BaseModel
from typing import List
import json 

from celery import Celery

from app.agents.conversation_model import Chat


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    input: List[Message]

def req_processing(chat, msg):
    messages = msg["input"]

    if len(messages)>=5:
        history = [m["content"] for m in messages[-5:-1]]
    else:
        history = [m["content"] for m in messages[:-1]]

    user_input = messages[-1]["content"]
    response = json.dumps(chat.run(user_input, history))

    return response

app = Celery(
    "worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
    )

app.conf.result_expires = 900

chat = Chat()

@app.task
def process_msg(messages):
    return req_processing(chat, messages)