"""
Routine Prediction Transformer Model
Simplified version for production
"""
from typing import Dict, List

import torch
import torch.nn as nn


class RoutinePredictor(nn.Module):
    """Simple routine predictor"""

    def __init__(self, vocab_size: int = 100, d_model: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Simple embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # Average pooling
        logits = self.fc(x)
        return logits

    def predict(self, activities: List[str], top_k: int = 5) -> List[Dict]:
        """Predict next activities"""
        # Demo implementation - returns mock predictions
        predictions = [
            {"activity": "Exercise", "confidence": 0.87},
            {"activity": "Coffee Break", "confidence": 0.82},
            {"activity": "Check Email", "confidence": 0.76},
            {"activity": "Meeting", "confidence": 0.71},
            {"activity": "Lunch", "confidence": 0.68},
        ]
        return predictions[:top_k]


# Initialize global model
routine_model = RoutinePredictor()
