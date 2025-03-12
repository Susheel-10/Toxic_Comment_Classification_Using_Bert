from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
from src.preprocessing.text_processor import TextPreprocessor
from src.models.toxic_classifier import ToxicClassifier

app = FastAPI()

class CommentRequest(BaseModel):
    text: str

class ToxicityResponse(BaseModel):
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float
    confidence: float

@app.post("/predict", response_model=ToxicityResponse)
async def predict_toxicity(comment: CommentRequest):
    try:
        # Preprocess text
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.process(comment.text)
        
        # Tokenize for BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            outputs = model(
                encoded['input_ids'].to(device),
                encoded['attention_mask'].to(device)
            )
        
        predictions = outputs[0].cpu().numpy()
        confidence = float(outputs.max())
        
        return ToxicityResponse(
            toxic=float(predictions[0]),
            severe_toxic=float(predictions[1]),
            obscene=float(predictions[2]),
            threat=float(predictions[3]),
            insult=float(predictions[4]),
            identity_hate=float(predictions[5]),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 