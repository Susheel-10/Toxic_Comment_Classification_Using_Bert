import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple

class ToxicClassifier(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super(ToxicClassifier, self).__init__()
        
        # BERT base model - freeze some layers to prevent overfitting
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        # Freeze the first 8 layers of BERT
        for param in list(self.bert.parameters())[:-8]:
            param.requires_grad = False
        
        # Simplified architecture focusing on BERT's power
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)  # 768 is BERT's hidden size
        
        # Initialize the classifier weights properly
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get BERT embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits  # Return logits directly, BCEWithLogitsLoss will handle the sigmoid