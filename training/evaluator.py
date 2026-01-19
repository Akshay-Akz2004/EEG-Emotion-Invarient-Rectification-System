# training/evaluator.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from config import config

class EPRNEvaluator:
    """Evaluator class for EPRN model"""
    
    def __init__(self, model, device=None):
        self.model = model
        if device is None:
            self.device = config.DEVICE
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        test_subject_preds, test_subject_labels = [], []
        test_emotion_preds, test_emotion_labels = [], []
        
        with torch.no_grad():
            for data, subject_labels, emotion_labels in test_loader:
                data = data.to(self.device)
                subject_logits, emotion_logits = self.model(data)
                
                subject_preds = subject_logits.argmax(dim=1).cpu().numpy()
                emotion_preds = emotion_logits.argmax(dim=1).cpu().numpy()
                
                test_subject_preds.extend(subject_preds)
                test_subject_labels.extend(subject_labels.numpy())
                test_emotion_preds.extend(emotion_preds)
                test_emotion_labels.extend(emotion_labels.numpy())
        
        test_acc = accuracy_score(test_subject_labels, test_subject_preds)
        test_emo_acc = accuracy_score(test_emotion_labels, test_emotion_preds)
        
        return test_acc, test_emo_acc, test_subject_preds, test_subject_labels
    
    def evaluate_per_emotion(self, test_loader, dataset, test_idx):
        """Evaluate model performance per emotion"""
        # Get emotion labels for test set
        test_emotion_labels_full = dataset.emotion_labels[test_idx]
        
        test_subject_preds, test_subject_labels = [], []
        with torch.no_grad():
            for data, subject_labels, _ in test_loader:
                data = data.to(self.device)
                subject_logits, _ = self.model(data)
                subject_preds = subject_logits.argmax(dim=1).cpu().numpy()
                test_subject_preds.extend(subject_preds)
                test_subject_labels.extend(subject_labels.numpy())
        
        test_subject_preds = np.array(test_subject_preds)
        test_subject_labels = np.array(test_subject_labels)
        
        emotion_accs = []
        emotion_samples = []
        
        for emotion_id in range(config.NUM_EMOTIONS):
            emotion_mask = (test_emotion_labels_full == emotion_id)
            
            if emotion_mask.sum() > 0:
                emotion_preds = test_subject_preds[emotion_mask]
                emotion_true = test_subject_labels[emotion_mask]
                
                emotion_acc = accuracy_score(emotion_true, emotion_preds)
                emotion_accs.append(emotion_acc)
                emotion_samples.append(emotion_mask.sum())
                
                print(f"  {config.EMOTION_NAMES[emotion_id]}: {emotion_acc*100:.2f}% "
                      f"({emotion_mask.sum()} samples)")
        
        # Calculate statistics
        emotion_variance = np.std(emotion_accs) * 100
        emotion_mean = np.mean(emotion_accs) * 100
        max_diff = (max(emotion_accs) - min(emotion_accs)) * 100
        
        return emotion_accs, emotion_samples, emotion_variance, emotion_mean, max_diff
    
    def save_results(self, results_dict, history=None):
        """Save evaluation results"""
        os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
        
        # Save main results
        pd.DataFrame([results_dict]).to_csv(config.RESULTS_CSV, index=False)
        
        # Save per-emotion results
        if 'per_emotion_accs' in results_dict:
            per_emotion_df = pd.DataFrame({
                'emotion': config.EMOTION_NAMES,
                'accuracy': [acc*100 for acc in results_dict['per_emotion_accs']],
                'samples': results_dict['per_emotion_samples']
            })
            per_emotion_df.to_csv(config.PER_EMOTION_CSV, index=False)
        
        # Save training history
        if history is not None:
            history_df = pd.DataFrame(history)
            history_df.to_csv(config.HISTORY_CSV, index=False)
        
        print(f"\n✓ Results saved to {config.RESULTS_CSV}")
        if 'per_emotion_accs' in results_dict:
            print(f"✓ Per-emotion results saved to {config.PER_EMOTION_CSV}")
        if history is not None:
            print(f"✓ Training history saved to {config.HISTORY_CSV}")