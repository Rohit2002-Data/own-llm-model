from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv 
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")
app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", token=HF_TOKEN, torch_dtype=torch.float16,device_map="auto")

@app.post("/generate/")
async def generate(request: Request):
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    data = await request.json()
    prompt = data.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}
