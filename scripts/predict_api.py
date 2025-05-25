from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

app = FastAPI()
tokenizer = DistilBertTokenizerFast.from_pretrained("nlp/tokenizer/")
model = DistilBertForSequenceClassification.from_pretrained("model/")
model.eval()

class ChatInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: ChatInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()
    return {"prediction": prediction}
