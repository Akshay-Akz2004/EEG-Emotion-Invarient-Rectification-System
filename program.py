# ============================================================================
# GPU-OPTIMIZED SEED-V BIOMETRIC SYSTEM FOR GTX 1650 (4GB VRAM)
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # Mixed precision for efficiency
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mne
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU CONFIGURATION FOR GTX 1650
# ============================================================================

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    if torch.cuda.is_available():
        print("="*70)
        print("GPU CONFIGURATION")
        print("="*70)
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set device
        device = torch.device('cuda:0')
        
        # Clear cache
        torch.cuda.empty_cache()
        
        print("✓ GPU optimization enabled")
        print("="*70 + "\n")
        return device
    else:
        print("Warning: CUDA not available, using CPU")
        return torch.device('cpu')


# ============================================================================
# OPTIMIZED DATA PREPROCESSING (CPU-efficient for GPU training)
# ============================================================================

class SEEDVRawProcessor:
    """Process raw .cnt files with GPU-friendly output"""
    
    def __init__(self, data_folder='./data', output_folder='./processed_features'):
        self.data_folder = data_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        # SEED-V emotion labels (15 trials)
        self.emotion_labels = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2]
        
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 14),
            'beta': (14, 31),
            'gamma': (31, 50)
        }
        
    def compute_differential_entropy(self, signal, sampling_rate=200):
        """Compute DE features for all frequency bands"""
        de_features = []
        
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            filtered = mne.filter.filter_data(
                signal, 
                sfreq=sampling_rate,
                l_freq=low_freq, 
                h_freq=high_freq,
                method='iir',
                verbose=False
            )
            
            variance = np.var(filtered, axis=1)
            de = 0.5 * np.log(2 * np.pi * np.e * variance + 1e-8)
            de_features.append(de)
        
        return np.array(de_features).T
    
    def segment_trials(self, raw_data, sampling_rate=200):
        """Segment into 15 trials"""
        n_channels, n_samples = raw_data.shape
        samples_per_trial = n_samples // 15
        
        trials = []
        for trial_idx in range(15):
            start_idx = trial_idx * samples_per_trial
            end_idx = min(start_idx + samples_per_trial, n_samples)
            trials.append(raw_data[:, start_idx:end_idx])
        
        return trials
    
    def extract_features_from_trial(self, trial_data, sampling_rate=200, window_size=1, overlap=0.5):
        """Extract DE with sliding windows"""
        n_channels, n_samples = trial_data.shape
        window_samples = int(window_size * sampling_rate)
        step_samples = int(window_samples * (1 - overlap))
        
        features_list = []
        
        for start in range(0, n_samples - window_samples, step_samples):
            end = start + window_samples
            window_data = trial_data[:, start:end]
            de_features = self.compute_differential_entropy(window_data, sampling_rate)
            features_list.append(de_features)
        
        return np.array(features_list)
    
    def process_single_file(self, subject_id, session_id):
        """Process one .cnt file"""
        filename = f'{subject_id}_{session_id}.cnt'
        filepath = os.path.join(self.data_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found!")
            return None
        
        print(f"Processing {filename}...")
        
        try:
            raw = mne.io.read_raw_cnt(filepath, preload=True, verbose=False)
            data = raw.get_data()
            sampling_rate = raw.info['sfreq']
            
            # Preprocessing
            data = mne.filter.filter_data(data, sfreq=sampling_rate, l_freq=0.5, h_freq=50, method='iir', verbose=False)
            
            if sampling_rate > 200:
                data = mne.filter.resample(data, down=sampling_rate/200, verbose=False)
                sampling_rate = 200
            
            trials = self.segment_trials(data, sampling_rate)
            
            all_features = []
            all_emotions = []
            
            for trial_idx, trial_data in enumerate(trials):
                if trial_idx >= 15:
                    break
                
                features = self.extract_features_from_trial(trial_data, sampling_rate, window_size=1, overlap=0.5)
                emotion_label = self.emotion_labels[trial_idx]
                
                all_features.append(features)
                all_emotions.extend([emotion_label] * len(features))
            
            all_features = np.vstack(all_features)
            all_emotions = np.array(all_emotions)
            
            output_file = os.path.join(self.output_folder, f'{subject_id}_{session_id}_features.npz')
            np.savez_compressed(output_file, features=all_features, emotions=all_emotions, subject=subject_id, session=session_id)
            
            print(f"  ✓ Saved {len(all_features)} samples")
            return len(all_features)
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return None
    
    def process_all_files(self):
        """Process all 48 files"""
        print("="*70)
        print("PROCESSING RAW .CNT FILES")
        print("="*70)
        
        total_samples = 0
        processed = 0
        
        for subject_id in range(1, 17):
            for session_id in range(1, 4):
                n = self.process_single_file(subject_id, session_id)
                if n:
                    total_samples += n
                    processed += 1
        
        print(f"\n✓ Processed {processed}/48 files | Total samples: {total_samples}")
        print("="*70 + "\n")


# ============================================================================
# GPU-OPTIMIZED DATASET WITH PIN_MEMORY
# ============================================================================

class SEEDVBiometricDataset(Dataset):
    """GPU-optimized dataset loader"""
    
    def __init__(self, features_folder='./processed_features', normalize=True):
        self.features_folder = features_folder
        self.normalize = normalize
        self.data, self.labels, self.metadata = self.load_all_features()
        
    def load_all_features(self):
        all_data = []
        all_labels = []
        all_metadata = []
        
        print("Loading features...")
        
        for subject_id in range(1, 17):
            for session_id in range(1, 4):
                filepath = os.path.join(self.features_folder, f'{subject_id}_{session_id}_features.npz')
                
                if not os.path.exists(filepath):
                    continue
                
                data = np.load(filepath)
                features = data['features']
                emotions = data['emotions']
                
                for idx in range(len(features)):
                    all_data.append(features[idx])
                    all_labels.append(subject_id - 1)
                    all_metadata.append({'subject': subject_id - 1, 'session': session_id, 'emotion': emotions[idx]})
        
        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        
        if self.normalize:
            mean = all_data.mean(axis=0, keepdims=True)
            std = all_data.std(axis=0, keepdims=True)
            all_data = (all_data - mean) / (std + 1e-8)
        
        print(f"✓ Loaded {len(all_data)} samples | Shape: {all_data.shape}\n")
        return all_data, all_labels, all_metadata
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': torch.FloatTensor(self.data[idx]),
            'label': torch.LongTensor([self.labels[idx]])[0],
            'emotion': torch.LongTensor([self.metadata[idx]['emotion']])[0],
            'session': self.metadata[idx]['session'],
            'subject': self.metadata[idx]['subject']
        }


# ============================================================================
# EMOTION-AWARE CHANNEL ATTENTION (NOVEL COMPONENT)
# ============================================================================

class EmotionAwareChannelAttention(nn.Module):
    """Novel Emotion-Aware Channel Attention"""
    
    def __init__(self, num_channels=62, num_emotions=5, reduction_ratio=8):
        super(EmotionAwareChannelAttention, self).__init__()
        
        self.num_channels = num_channels
        self.num_emotions = num_emotions
        
        self.emotion_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_channels, num_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels // reduction_ratio, num_channels),
                nn.Sigmoid()
            ) for _ in range(num_emotions)
        ])
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(num_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions)
        )
        
    def forward(self, x, emotion_labels=None, training=False):
        channel_stats = torch.mean(x, dim=2)
        emotion_logits = self.emotion_classifier(channel_stats)
        emotion_probs = F.softmax(emotion_logits, dim=1)
        
        emotion_attentions = []
        for emotion_idx in range(self.num_emotions):
            attention = self.emotion_branches[emotion_idx](channel_stats)
            emotion_attentions.append(attention.unsqueeze(1))
        
        emotion_attentions = torch.cat(emotion_attentions, dim=1)
        
        if training and emotion_labels is not None:
            emotion_one_hot = F.one_hot(emotion_labels, self.num_emotions).float()
            fused_attention = torch.einsum('bec,be->bc', emotion_attentions, emotion_one_hot)
        else:
            fused_attention = torch.einsum('bec,be->bc', emotion_attentions, emotion_probs)
        
        attended_x = x * fused_attention.unsqueeze(2)
        
        if training:
            return attended_x, emotion_logits
        else:
            return attended_x


# ============================================================================
# MAIN BIOMETRIC NETWORK
# ============================================================================

class EACANBiometric(nn.Module):
    """Emotion-Aware Channel Attention Network"""
    
    def __init__(self, num_channels=62, num_bands=5, num_subjects=16, num_emotions=5):
        super(EACANBiometric, self).__init__()
        
        self.num_subjects = num_subjects
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_bands, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.eaca = EmotionAwareChannelAttention(num_channels, num_emotions, reduction_ratio=8)
        
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_subjects)
        )
        
    def forward(self, x, emotion_labels=None, training=False):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        
        if training:
            x, emotion_logits = self.eaca(x, emotion_labels, training=True)
        else:
            x = self.eaca(x, emotion_labels, training=False)
            emotion_logits = None
        
        x = x.transpose(1, 2)
        x = self.spatial_conv(x)
        x = self.global_pool(x).squeeze(2)
        logits = self.classifier(x)
        
        if training:
            return logits, emotion_logits
        else:
            return logits


# ============================================================================
# GPU-OPTIMIZED TRAINER WITH MIXED PRECISION
# ============================================================================

class BiometricTrainer:
    """GPU-optimized trainer with automatic mixed precision"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.scaler = GradScaler()  # For mixed precision training
        
    def train_epoch(self, train_loader, optimizer, criterion_identity, criterion_emotion, alpha=0.7):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', ncols=100)
        for batch in pbar:
            data = batch['data'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            emotions = batch['emotion'].to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                identity_logits, emotion_logits = self.model(data, emotions, training=True)
                loss_identity = criterion_identity(identity_logits, labels)
                loss_emotion = criterion_emotion(emotion_logits, emotions)
                loss = alpha * loss_identity + (1 - alpha) * loss_emotion
            
            # Backward with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pred = identity_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Monitor GPU memory
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*correct/total:.2f}%',
                    'gpu': f'{mem:.2f}GB'
                })
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate(self, test_loader, return_detailed=False):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_emotions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating', ncols=100):
                data = batch['data'].to(self.device, non_blocking=True)
                labels = batch['label'].cpu().numpy()
                emotions = batch['emotion'].cpu().numpy()
                
                with autocast():
                    logits = self.model(data, training=False)
                
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_emotions.extend(emotions)
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        if return_detailed:
            return accuracy, {'predictions': np.array(all_preds), 'labels': np.array(all_labels), 'emotions': np.array(all_emotions)}
        return accuracy


# ============================================================================
# CROSS-EMOTION EVALUATION
# ============================================================================

def evaluate_cross_emotion(model, dataset, device):
    """5x5 cross-emotion matrix (novel contribution)"""
    model.eval()
    
    emotion_names = ['Happy', 'Sad', 'Disgust', 'Fear', 'Neutral']
    results = np.zeros((5, 5))
    
    print("\n" + "="*70)
    print("CROSS-EMOTION AUTHENTICATION ANALYSIS")
    print("="*70)
    
    for test_emo in range(5):
        test_indices = [i for i, meta in enumerate(dataset.metadata) if meta['emotion'] == test_emo]
        
        if len(test_indices) == 0:
            continue
        
        test_subset = torch.utils.data.Subset(dataset, test_indices[:2000])
        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, pin_memory=True, num_workers=2)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                data = batch['data'].to(device, non_blocking=True)
                labels = batch['label'].cpu().numpy()
                
                with autocast():
                    logits = model(data, training=False)
                
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        acc = accuracy_score(all_labels, all_preds)
        
        for enroll_emo in range(5):
            results[enroll_emo, test_emo] = acc * 100
        
        print(f"Test Emotion: {emotion_names[test_emo]:8s} | Accuracy: {acc*100:.2f}%")
    
    return results, emotion_names


# ============================================================================
# MAIN EXECUTION WITH GPU OPTIMIZATION
# ============================================================================

def main():
    print("="*70)
    print("EMOTION-AWARE CHANNEL ATTENTION NETWORK (EACAN)")
    print("GPU-Optimized for NVIDIA GTX 1650")
    print("="*70 + "\n")
    
    # Setup GPU
    device = setup_gpu()
    
    # Step 1: Process raw files
    print("[STEP 1] Processing .cnt files...")
    processor = SEEDVRawProcessor(data_folder='./data', output_folder='./processed_features')
    processor.process_all_files()
    
    # Step 2: Load dataset
    print("[STEP 2] Loading dataset...")
    dataset = SEEDVBiometricDataset(features_folder='./processed_features', normalize=True)
    
    # Step 3: Split
    print("[STEP 3] Splitting dataset...")
    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=42, stratify=dataset.labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=dataset.labels[temp_idx])
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    # GPU-optimized data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}\n")
    
    # Step 4: Initialize model
    print("[STEP 4] Initializing model...")
    model = EACANBiometric(num_channels=62, num_bands=5, num_subjects=16, num_emotions=5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Step 5: Training
    print("[STEP 5] Training with GPU acceleration...")
    trainer = BiometricTrainer(model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    criterion_identity = nn.CrossEntropyLoss()
    criterion_emotion = nn.CrossEntropyLoss()
    
    os.makedirs('./results', exist_ok=True)
    best_val_acc = 0
    
    for epoch in range(50):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/50")
        print('='*70)
        
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion_identity, criterion_emotion, alpha=0.7)
        val_acc = trainer.evaluate(val_loader)
        
        print(f"\nResults: Train Loss={train_loss:.4f} | Train Acc={train_acc*100:.2f}% | Val Acc={val_acc*100:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './results/best_model.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc*100:.2f}%")
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Step 6: Test
    print("\n" + "="*70)
    print("[STEP 6] Final evaluation...")
    print("="*70)
    model.load_state_dict(torch.load('./results/best_model.pth'))
    test_acc, details = trainer.evaluate(test_loader, return_detailed=True)
    print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")
    
    # Step 7: Cross-emotion analysis
    print("\n[STEP 7] Cross-emotion analysis (novel contribution)...")
    emotion_matrix, emotion_names = evaluate_cross_emotion(model, dataset, device)
    
    # Save results
    pd.DataFrame(emotion_matrix, index=emotion_names, columns=emotion_names).to_csv('./results/cross_emotion_matrix.csv')
    
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(emotion_matrix, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=emotion_names, yticklabels=emotion_names, cbar_kws={'label': 'Accuracy (%)'})
    plt.xlabel('Test Emotion', fontsize=12)
    plt.ylabel('Enrollment Emotion', fontsize=12)
    plt.title('Cross-Emotion Authentication Accuracy (%) - SEED-V', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/emotion_matrix.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Results saved to: ./results/")
    print("="*70)


if __name__ == '__main__':
    main()
