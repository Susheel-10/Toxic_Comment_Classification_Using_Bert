import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm
from torch.amp import autocast, GradScaler

class ModelTrainer:
    def __init__(self, model, optimizer, criterion, device, scaler: GradScaler = None, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = scaler or GradScaler('cuda')
        self.use_amp = device.type == 'cuda'
        self.scheduler = scheduler

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()

        return {'loss': total_loss / len(dataloader)}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(input_ids, attention_mask)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                
                # Apply sigmoid to get probabilities for predictions
                probs = torch.sigmoid(outputs)
                
                total_loss += loss.item()
                predictions.extend(probs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        return {
            'loss': total_loss / len(dataloader),
            'predictions': predictions,
            'true_labels': true_labels
        } 