# model/loss.py
import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    """Combined loss with gradient balancing"""
    def __init__(self, lambda_emotion=None, label_smoothing=None):
        super().__init__()
        from config import config
        
        if lambda_emotion is None:
            lambda_emotion = config.LAMBDA_EMOTION
        if label_smoothing is None:
            label_smoothing = config.LABEL_SMOOTHING
            
        self.lambda_emotion = lambda_emotion
        self.subject_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, subject_logits, emotion_logits, subject_labels, emotion_labels):
        subject_loss = self.subject_criterion(subject_logits, subject_labels)
        emotion_loss = self.emotion_criterion(emotion_logits, emotion_labels)
        
        # Total loss
        total_loss = subject_loss + self.lambda_emotion * emotion_loss
        
        return total_loss, subject_loss, emotion_loss