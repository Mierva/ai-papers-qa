import sys
import os

# Add the project root (ai-papers-qa) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_model.model import LlamaModel

app = FastAPI()

# llama = LlamaModel()
# llama.init()
# print('\nLoaded model successfuly.\n')
generated_outputs = {}

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100

llama = None  # Start with an uninitialized model

@app.on_event("startup")
async def load_model():
    global llama
    if llama is None:
        llama = LlamaModel()
        llama.init()
        print('\nLoaded model successfully.\n')


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    global llama
    if llama is None:
        llama = LlamaModel()
        llama.init()
        print('\nLoaded model successfully.\n')
    output = llama.generate(prompt=request.prompt, max_length=request.max_length)
    return {"output_id": len(generated_outputs) + 1, "output": output}


@app.get("/output/{output_id}")
async def get_output(output_id: int):
    output = generated_outputs.get(output_id)
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")
    return {"output_id": output_id, "output": output}
