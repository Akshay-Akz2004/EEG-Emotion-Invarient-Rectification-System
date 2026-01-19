# model/eprn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class RobustEPRN(nn.Module):
    """EPRN with strong regularization"""
    def __init__(self, num_subjects=None, num_emotions=None):
        super().__init__()
        
        if num_subjects is None:
            num_subjects = config.NUM_SUBJECTS
        if num_emotions is None:
            num_emotions = config.NUM_EMOTIONS
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(config.INPUT_CHANNELS, config.ENCODER_CHANNELS[0], 3, padding=1),
            nn.BatchNorm1d(config.ENCODER_CHANNELS[0]),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Conv1d(config.ENCODER_CHANNELS[0], config.ENCODER_CHANNELS[1], 3, padding=1),
            nn.BatchNorm1d(config.ENCODER_CHANNELS[1]),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Conv1d(config.ENCODER_CHANNELS[1], config.ENCODER_CHANNELS[2], 3, padding=1),
            nn.BatchNorm1d(config.ENCODER_CHANNELS[2]),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Emotion recognition head
        self.emotion_head = nn.Sequential(
            nn.Linear(config.ENCODER_CHANNELS[2], config.EMOTION_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.EMOTION_HEAD_HIDDEN, num_emotions)
        )
        
        # Emotion prototype generators
        self.prototype_generators = nn.ModuleList([
            nn.Linear(config.ENCODER_CHANNELS[2], config.ENCODER_CHANNELS[2])
            for _ in range(num_emotions)
        ])
        
        # Subject classifier
        self.subject_classifier = nn.Sequential(
            nn.Linear(config.ENCODER_CHANNELS[2], config.SUBJECT_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.SUBJECT_HEAD_HIDDEN, num_subjects)
        )
        
        # Rectification coefficient
        self.register_buffer('rectification_alpha', torch.tensor(config.RECTIFICATION_ALPHA))
        
        # Layer normalization for stabilization
        self.layer_norm = nn.LayerNorm(config.ENCODER_CHANNELS[2])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass - returns (subject_logits, emotion_logits, rectified_features)"""
        # x shape: (batch, 66, 5) -> (batch, 5, 66)
        x = x.transpose(1, 2)
        
        # Extract features
        features = self.encoder(x).squeeze(2)  # (batch, 128)
        
        # Add small noise during training
        if self.training:
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        # Emotion recognition
        emotion_logits = self.emotion_head(features)
        emotion_probs = F.softmax(emotion_logits, dim=1)
        
        # Generate weighted emotion prototype
        prototype_weights = []
        for generator in self.prototype_generators:
            proto = generator(features)
            prototype_weights.append(proto.unsqueeze(1))
        
        prototypes = torch.cat(prototype_weights, dim=1)  # (batch, num_emotions, 128)
        
        # Weighted average using emotion probabilities
        weighted_prototype = torch.sum(
            emotion_probs.unsqueeze(2) * prototypes,
            dim=1
        )  # (batch, 128)
        
        # Rectification (as per EPRN paper)
        rectified_features = features - self.rectification_alpha * weighted_prototype
        
        # Layer normalization for stability
        rectified_features = self.layer_norm(rectified_features)
        
        # Subject classification
        subject_logits = self.subject_classifier(rectified_features)
        
        return subject_logits, emotion_logits, rectified_features
    
    def extract_features(self, x):
        """Extract only the rectified features (for embeddings)"""
        subject_logits, emotion_logits, rectified_features = self.forward(x)
        return rectified_features