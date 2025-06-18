from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv 

load_dotenv()

API_KEY = os.getenv("API_KEY")

app = FastAPI()

# ✅ Use lightweight distilgpt2 without token
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cpu")  # ✅ Force CPU

@app.post("/generate/")
async def generate(request: Request):
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    data = await request.json()
    prompt = data.get("prompt", "").strip()
    
    if not prompt:
        return {"response": "Prompt is empty."}

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": result}

