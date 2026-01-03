"""
Production Routine Predictor - Full Transformer Implementation
"""
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionalEncoding(nn.Module):
    """Positional encoding with temporal awareness"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Standard positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Temporal encodings
        self.hour_embedding = nn.Embedding(24, d_model // 4)
        self.dow_embedding = nn.Embedding(7, d_model // 4)
        self.month_embedding = nn.Embedding(12, d_model // 4)

    def forward(
        self,
        x: torch.Tensor,
        hour: torch.Tensor,
        day_of_week: torch.Tensor,
        month: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Standard positional encoding
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)
        pos_enc = pos_enc.expand(batch_size, -1, -1)

        # Temporal embeddings
        hour_enc = self.hour_embedding(hour)
        dow_enc = self.dow_embedding(day_of_week)
        month_enc = self.month_embedding(month)

        temporal_enc = torch.cat([hour_enc, dow_enc, month_enc], dim=-1)
        temporal_enc = temporal_enc.unsqueeze(1).expand(-1, seq_len, -1)

        # Pad to match d_model
        pad_size = self.d_model - temporal_enc.shape[-1]
        temporal_enc = F.pad(temporal_enc, (0, pad_size))

        return x + pos_enc + temporal_enc


class RoutineTransformerFull(nn.Module):
    """Full Transformer for routine prediction"""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = TemporalPositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.confidence_head = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        hour: torch.Tensor,
        day_of_week: torch.Tensor,
        month: torch.Tensor,
    ):
        # Embed
        x = self.embedding(input_ids) * np.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x, hour, day_of_week, month)

        # Transform
        encoded = self.transformer(x)

        # Output
        logits = self.output_projection(encoded)
        confidence = torch.sigmoid(self.confidence_head(encoded))

        return logits, confidence

    def predict_next(
        self,
        activities: List[str],
        hour: int,
        day_of_week: int,
        month: int,
        top_k: int = 5,
    ) -> List[Dict]:
        """Predict next activities"""
        self.eval()

        # Mock vocabulary for demo
        activity_vocab = {
            "wake_up": 0,
            "coffee": 1,
            "exercise": 2,
            "shower": 3,
            "breakfast": 4,
            "commute": 5,
            "work": 6,
            "lunch": 7,
            "meeting": 8,
            "email": 9,
            "coding": 10,
        }

        # Convert activities to IDs
        activity_ids = [activity_vocab.get(a.lower(), 0) for a in activities]

        if not activity_ids:
            activity_ids = [0]

        # Prepare inputs
        input_ids = torch.tensor([activity_ids], dtype=torch.long)
        hour_t = torch.tensor([hour], dtype=torch.long)
        dow_t = torch.tensor([day_of_week], dtype=torch.long)
        month_t = torch.tensor([month], dtype=torch.long)

        with torch.no_grad():
            logits, confidence = self.forward(input_ids, hour_t, dow_t, month_t)

            # Get predictions for last position
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits, dim=0)

            top_probs, top_indices = torch.topk(probs, k=top_k)

            # Reverse vocabulary
            id_to_activity = {v: k for k, v in activity_vocab.items()}

            predictions = []
            for i in range(top_k):
                activity_id = top_indices[i].item()
                activity_name = id_to_activity.get(
                    activity_id, f"activity_{activity_id}"
                )
                predictions.append(
                    {
                        "activity": activity_name.replace("_", " ").title(),
                        "confidence": float(top_probs[i]),
                    }
                )

        return predictions


# Initialize model
routine_transformer = RoutineTransformerFull(
    vocab_size=100, d_model=128, nhead=4, num_layers=2
)
