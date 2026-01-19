# data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from config import config

class EPRNSimpleDataset(Dataset):
    """Simple EPRN dataset with emotion labels"""
    def __init__(self, folder=None):
        if folder is None:
            folder = config.DATA_FOLDER
        print("Loading EEG data...")
        
        all_data, all_subject_labels, all_emotion_labels = [], [], []
        
        # Load all files
        for subj in range(1, config.NUM_SUBJECTS + 1):
            for sess in range(1, 4):
                path = f"{folder}/{subj}_{sess}_features.npz"
                if os.path.exists(path):
                    data = np.load(path)
                    features = data['features']
                    
                    # Take first samples per file
                    n_samples = min(config.SAMPLE_PER_FILE, len(features))
                    for i in range(n_samples):
                        if features[i].shape == (config.INPUT_LENGTH, config.INPUT_CHANNELS):
                            all_data.append(features[i])
                            all_subject_labels.append(subj - 1)  # 0-indexed
                            
                            # Assign emotion: SEED-V has 4 emotions
                            emotion_idx = (i // (n_samples // 4)) % 4
                            all_emotion_labels.append(emotion_idx)
        
        self.data = np.array(all_data, dtype=np.float32)
        self.subject_labels = np.array(all_subject_labels, dtype=np.int64)
        self.emotion_labels = np.array(all_emotion_labels, dtype=np.int64)
        
        # Simple normalization
        self.data = (self.data - self.data.mean()) / (self.data.std() + 1e-8)
        
        print(f"Loaded {len(self.data):,} samples")
        print(f"Shape: {self.data.shape}")
        print(f"Subjects: {len(np.unique(self.subject_labels))}")
        print(f"Emotions: {len(np.unique(self.emotion_labels))}")
        
        # Check distribution
        unique_subj, counts_subj = np.unique(self.subject_labels, return_counts=True)
        unique_emo, counts_emo = np.unique(self.emotion_labels, return_counts=True)
        print(f"Subject distribution: {dict(zip(unique_subj, counts_subj))}")
        print(f"Emotion distribution: {dict(zip(unique_emo, counts_emo))}\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.data[idx]),  # (66, 5)
            torch.LongTensor([self.subject_labels[idx]])[0],
            torch.LongTensor([self.emotion_labels[idx]])[0]
        )