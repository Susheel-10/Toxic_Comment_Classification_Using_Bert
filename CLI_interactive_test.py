import torch
from test_model import ToxicPredictor
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load model
    MODEL_PATH = os.path.join("models", "saved", "best_model.pt")
    predictor = ToxicPredictor(MODEL_PATH)
    
    print("\n=== Toxic Comment Classifier ===")
    print("Enter 'quit' to exit")
    
    while True:
        # Get input
        print("\nEnter a comment to analyze:")
        comment = input("> ")
        
        if comment.lower() == 'quit':
            break
        
        # Get predictions
        predictions = predictor.predict(comment)
        
        # Print results
        print("\nResults:")
        print("-" * 50)
        for category, score in predictions.items():
            print(f"{category:12}: {score:.4f}")
        
        # Overall assessment
        max_toxicity = max(predictions.values())
        print("-" * 50)
        if max_toxicity > 0.5:
            print(f"⚠️  Warning: Comment may be toxic (max score: {max_toxicity:.4f})")
        else:
            print(f"✅  Comment appears non-toxic (max score: {max_toxicity:.4f})")

if __name__ == "__main__":
    main() 