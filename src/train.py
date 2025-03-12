import torch
from transformers import BertTokenizer, AdamW
from src.models.toxic_classifier import ToxicClassifier
from src.models.trainer import ModelTrainer
from src.data.data_loader import load_toxic_data, create_data_loaders
import logging
import os
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    data_path: str,
    model_save_path: str,
    num_epochs: int = 5,
    batch_size: int = 64,  # Increased for RTX 3060
    learning_rate: float = 2e-5,
    max_grad_norm: float = 1.0
):
    # Set device and enable CUDA optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    logger.info("Loading dataset...")
    texts, labels = load_toxic_data(data_path)
    train_loader, val_loader = create_data_loaders(
        texts, 
        labels, 
        tokenizer, 
        batch_size=batch_size
    )

    # Initialize model
    logger.info("Initializing model...")
    model = ToxicClassifier().to(device)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Initialize trainer with mixed precision support
    trainer = ModelTrainer(model, optimizer, criterion=torch.nn.BCELoss(), device=device, scaler=scaler)

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Training Loss: {train_metrics['loss']:.4f}")

        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        val_loss = val_metrics['loss']
        logger.info(f"Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(model_save_path, 'best_model.pt'))
            logger.info("Saved best model checkpoint")

    logger.info("Training completed!")

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "raw", "train.csv")
    MODEL_SAVE_PATH = os.path.join("models", "saved")
    
    # Create model save directory if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    train_model(DATA_PATH, MODEL_SAVE_PATH)