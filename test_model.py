import torch
from transformers import BertTokenizer
from src.models.toxic_classifier import ToxicClassifier
import logging
import os
import numpy as np
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ToxicPredictor:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = ToxicClassifier().to(self.device)
        
        # Load trained weights with weights_only=True for security
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Handle both old and new model state dict formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Load state dict and handle any missing/unexpected keys
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")
        
        self.model.eval()
        
        # Category names
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        logger.info("Model loaded successfully")

    def predict(self, text: str) -> Dict[str, float]:
        """Predict toxicity scores for a single text"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Create results dictionary
        results = {
            category: float(prob)
            for category, prob in zip(self.categories, probabilities)
        }
        
        return results

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict toxicity scores for a batch of texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

def test_model():
    # Path to your saved model
    MODEL_PATH = os.path.join("models", "saved", "best_model.pt")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        return
    
    # Initialize predictor
    predictor = ToxicPredictor(MODEL_PATH)
    
    # Test cases
    test_comments = [
        "You are a wonderful person!",
        "I hate you and everything you stand for!",
        "This is a neutral comment about the weather.",
        "You're an idiot and should die!",
        "Let's have a constructive discussion about this topic."
    ]
    
    # Make predictions
    logger.info("Testing model with example comments...")
    for comment in test_comments:
        predictions = predictor.predict(comment)
        
        logger.info("\nComment: " + comment)
        logger.info("Predictions:")
        for category, score in predictions.items():
            logger.info(f"{category}: {score:.4f}")
        
        # Overall toxicity assessment
        max_toxicity = max(predictions.values())
        if max_toxicity > 0.5:
            logger.info(f"⚠️ Comment may be toxic (max score: {max_toxicity:.4f})")
        else:
            logger.info(f"✅ Comment appears non-toxic (max score: {max_toxicity:.4f})")

if __name__ == "__main__":
    test_model() 