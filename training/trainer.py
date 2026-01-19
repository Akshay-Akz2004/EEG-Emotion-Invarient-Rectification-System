# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from config import config

class EPRNTrainer:
    """Trainer class for EPRN model"""
    
    def __init__(self, model, criterion, optimizer, scheduler, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if device is None:
            self.device = config.DEVICE
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
            'train_emo_acc': [], 'val_emo_acc': [], 'lr': []
        }
        
        self.best_val_acc = 0
        self.patience = 0
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        train_subject_correct = 0
        train_emotion_correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        for data, subject_labels, emotion_labels in pbar:
            data, subject_labels, emotion_labels = (
                data.to(self.device), 
                subject_labels.to(self.device), 
                emotion_labels.to(self.device)
            )
            
            self.optimizer.zero_grad()
            subject_logits, emotion_logits = self.model(data)
            
            # Calculate loss
            loss, subject_loss, emotion_loss = self.criterion(
                subject_logits, emotion_logits, 
                subject_labels, emotion_labels
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP)
            
            # Add small gradient noise (prevents overfitting)
            if epoch > 5:
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * 0.0001
                        param.grad.add_(noise)
            
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            
            # Subject accuracy
            subject_preds = subject_logits.argmax(dim=1)
            train_subject_correct += (subject_preds == subject_labels).sum().item()
            
            # Emotion accuracy
            emotion_preds = emotion_logits.argmax(dim=1)
            train_emotion_correct += (emotion_preds == emotion_labels).sum().item()
            
            total += subject_labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'subj_acc': f'{100*train_subject_correct/total:.1f}%',
                'emo_acc': f'{100*train_emotion_correct/total:.1f}%'
            })
        
        train_acc = train_subject_correct / total
        train_emo_acc = train_emotion_correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        return train_acc, train_emo_acc, avg_train_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_subject_preds, val_subject_labels = [], []
        val_emotion_preds, val_emotion_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            for data, subject_labels, emotion_labels in val_loader:
                data, subject_labels, emotion_labels = (
                    data.to(self.device), 
                    subject_labels.to(self.device), 
                    emotion_labels.to(self.device)
                )
                
                subject_logits, emotion_logits = self.model(data)
                
                # Calculate validation loss
                loss, _, _ = self.criterion(
                    subject_logits, emotion_logits,
                    subject_labels, emotion_labels
                )
                val_loss += loss.item()
                
                # Get predictions
                subject_preds = subject_logits.argmax(dim=1).cpu().numpy()
                emotion_preds = emotion_logits.argmax(dim=1).cpu().numpy()
                
                val_subject_preds.extend(subject_preds)
                val_subject_labels.extend(subject_labels.cpu().numpy())
                val_emotion_preds.extend(emotion_preds)
                val_emotion_labels.extend(emotion_labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(val_subject_labels, val_subject_preds)
        val_emo_acc = accuracy_score(val_emotion_labels, val_emotion_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        return val_acc, val_emo_acc, avg_val_loss, (val_subject_preds, val_subject_labels)
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS
        
        for epoch in range(num_epochs):
            # Training phase
            train_acc, train_emo_acc, train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_acc, val_emo_acc, val_loss, _ = self.validate(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Store history
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_emo_acc'].append(train_emo_acc)
            self.history['val_emo_acc'].append(val_emo_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Calculate accuracy gap (overfitting indicator)
            gap = (train_acc - val_acc) * 100
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}% | Gap: {gap:.1f}%")
            print(f"  Train Emo: {train_emo_acc*100:.2f}% | Val Emo: {val_emo_acc*100:.2f}%")
            print(f"  Loss: Train={train_loss:.3f} | Val={val_loss:.3f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience = 0
                self.save_checkpoint(epoch, train_acc, val_acc)
                print(f"  ✓ Saved best model (Val: {val_acc*100:.2f}%)")
            else:
                self.patience += 1
                print(f"  Patience: {self.patience}/{config.PATIENCE_LIMIT}")
                
                if self.patience >= config.PATIENCE_LIMIT:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    def save_checkpoint(self, epoch, train_acc, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, config.MODEL_SAVE_PATH)
    
    def load_checkpoint(self, path=None):
        """Load model checkpoint"""
        if path is None:
            path = config.MODEL_SAVE_PATH
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint