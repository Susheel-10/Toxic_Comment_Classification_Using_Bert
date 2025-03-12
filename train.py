import torch
from transformers import AutoTokenizer
from src.models.toxic_classifier import ToxicClassifier
from src.models.trainer import ModelTrainer
from src.data.data_loader import load_toxic_data, create_data_loaders
import logging
import os
from torch.amp import GradScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(
    data_path: str,
    model_save_path: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_grad_norm: float = 1.0
):
    try:
        # Set device and enable CUDA optimizations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        logger.info(f"Using device: {device}")

        # Load tokenizer
        logger.info("Loading BERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Load data
        logger.info("Loading and preparing dataset...")
        texts, labels = load_toxic_data(data_path)
        logger.info(f"Dataset size: {len(texts)} samples")
        
        # Calculate class weights for balanced loss
        pos_counts = labels.sum(axis=0)
        neg_counts = len(labels) - pos_counts
        pos_weights = torch.tensor(neg_counts / pos_counts).to(device)
        logger.info(f"Positive class weights: {pos_weights}")
        
        train_loader, val_loader = create_data_loaders(
            texts, 
            labels, 
            tokenizer, 
            batch_size=batch_size,
            num_workers=2
        )
        logger.info("Data loaders created successfully")

        # Initialize model
        logger.info("Initializing model...")
        model = ToxicClassifier(dropout=0.2).to(device)
        
        # Initialize optimizer with different learning rates for BERT and classifier
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.bert.named_parameters() if p.requires_grad],
                'lr': learning_rate / 10,
                'weight_decay': 0.01
            },
            {
                'params': model.classifier.parameters(),
                'lr': learning_rate,
                'weight_decay': 0.01
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        # Use BCEWithLogitsLoss with calculated positive weights
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler('cuda') if device.type == 'cuda' else None

        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler
        )
        logger.info("Model and trainer initialized successfully")

        # Training loop
        logger.info("Starting training...")
        best_val_loss = float('inf')
        patience = 3
        no_improve = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            logger.info(f"Training Loss: {train_metrics['loss']:.4f}")

            # Evaluate
            val_metrics = trainer.evaluate(val_loader)
            val_loss = val_metrics['loss']
            logger.info(f"Validation Loss: {val_loss:.4f}")

            # Save best model and check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                checkpoint_path = os.path.join(model_save_path, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best model checkpoint (loss: {best_val_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info("Training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    # Get the absolute path to the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths relative to project root
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "train.csv")
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "saved")
    
    # Verify data file exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        exit(1)
    
    # Create model save directory if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Print CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA is not available. Using CPU for training.")
    
    # Start training
    train_model(DATA_PATH, MODEL_SAVE_PATH)