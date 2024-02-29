from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
import asyncio

class Item(BaseModel):
    text: str

app = FastAPI()

async def get_classifier():
    return await asyncio.get_event_loop().run_in_executor(None, pipeline, "sentiment-analysis")

@app.on_event("startup")
async def on_startup():
    app.classifier = await get_classifier()

@app.get("/")
async def root():
    return {"FastApi service started!"}

@app.get("/{text}")
async def get_params(text: str):
    try:
        return await app.classifier(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to classify text")

@app.post("/predict/")
async def predict(item: Item):
    try:
        return await app.classifier(item.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to classify text")
