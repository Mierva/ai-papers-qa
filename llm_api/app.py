import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from llm_model.model import LlamaModel
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage
from llm_model.agent import Chat


app = FastAPI()
generated_outputs = {}
llama = None
conversation_chain = None

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100

@app.on_event("startup")
async def load_model():
    global llama
    if llama is None:
        llama = LlamaModel()
        llama._initialize_model()
        print('\nLoaded model successfully.\n')


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    output = llama._call([HumanMessage(content=request.prompt)])
    return {"output_id": len(generated_outputs) + 1, "output": output.generations}


@app.get("/output/{output_id}")
async def get_output(output_id: int):
    output = generated_outputs.get(output_id)
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")
    return {"output_id": output_id, "output": output}


@app.post('/chat')
async def chat(request: GenerateRequest):
    global conversation_chain
    chat = Chat(llm=llama, conversation_chain=conversation_chain)
    response = chat.get_message(request.prompt)
    conversation_chain = chat.get_conversation_chain()

    return {"response": response}