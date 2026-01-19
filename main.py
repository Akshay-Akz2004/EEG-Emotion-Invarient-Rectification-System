# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import random
import numpy as np
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data.dataset import EPRNSimpleDataset
from data.data_loader import create_data_loaders
from model.eprn_model import RobustEPRN
from model.loss import CombinedLoss
from training.trainer import EPRNTrainer
from training.evaluator import EPRNEvaluator
from utils.metrics import calculate_metrics, print_results_summary

def set_seeds(seed=None):
    """Set random seeds for reproducibility"""
    if seed is None:
        seed = config.SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_optimizer_scheduler(model):
    """Setup optimizer and scheduler"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.BETA1, config.BETA2),
        eps=1e-8
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.STEP_SIZE,
        gamma=config.GAMMA
    )
    
    return optimizer, scheduler

def print_success_message(test_acc, emotion_variance):
    """Print success/failure message based on results"""
    if test_acc >= 0.85:
        print(f"\n🎉 EXCELLENT! EPRN achieves {test_acc*100:.2f}% accuracy!")
        print(f"   Emotion variance: {emotion_variance:.2f}% (should be <2% for invariance)")
        print("\nYou can report in your paper:")
        print(f"'EPRN achieves {test_acc*100:.2f}% authentication accuracy with {emotion_variance:.2f}% variance across emotions'")
    else:
        print(f"\n⚠ Current accuracy: {test_acc*100:.2f}%")
        print("Try these quick fixes:")
        print("1. Train for 50 epochs instead of 40")
        print("2. Reduce learning rate to 0.0003")
        print("3. Increase batch size to 128")

def print_troubleshooting():
    """Print troubleshooting information"""
    print("\nTroubleshooting:")
    print(f"1. Ensure data files exist in '{config.DATA_FOLDER}/'")
    print("2. Expected file format: '1_1_features.npz' for subject 1, session 1")
    print("3. Each .npz file should contain 'features' array of shape (n_samples, 66, 5)")
    print("\nTo create dummy data for testing:")
    print("```python")
    print("import numpy as np")
    print("os.makedirs('./processed_features', exist_ok=True)")
    print("for subj in range(1, 17):")
    print("    for sess in range(1, 4):")
    print("        data = np.random.randn(5000, 66, 5).astype(np.float32)")
    print("        np.savez(f'./processed_features/{subj}_{sess}_features.npz', features=data)")
    print("```")

def main():
    """Main training and evaluation pipeline"""
    print("="*80)
    print("EEG Biometric Authentication with Emotion-Invariant Features")
    print("="*80)
    
    # Set random seeds
    set_seeds()
    
    # Check device
    print(f"Using device: {config.DEVICE}\n")
    
    # Create results directory
    os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
    
    # 1. Load dataset
    print("[1] Loading dataset...")
    try:
        dataset = EPRNSimpleDataset(config.DATA_FOLDER)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print_troubleshooting()
        return
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("Error: No data loaded!")
        print_troubleshooting()
        return
    
    # 2. Create data loaders
    print("\n[2] Creating data loaders...")
    try:
        train_loader, val_loader, test_loader, split_indices = create_data_loaders(dataset)
        train_idx, val_idx, test_idx = split_indices
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # 3. Initialize model
    print("\n[3] Initializing EPRN model...")
    model = RobustEPRN(config.NUM_SUBJECTS, config.NUM_EMOTIONS)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # 4. Setup loss, optimizer, and scheduler
    criterion = CombinedLoss(config.LAMBDA_EMOTION, config.LABEL_SMOOTHING)
    optimizer, scheduler = setup_optimizer_scheduler(model)
    
    # 5. Create trainer and train
    print("[4] Starting training...")
    print("="*80)
    
    trainer = EPRNTrainer(model, criterion, optimizer, scheduler, config.DEVICE)
    
    try:
        history = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Load best model and evaluate
    print("\n[5] Final evaluation with best model...")
    
    try:
        checkpoint = trainer.load_checkpoint(config.MODEL_SAVE_PATH)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {config.MODEL_SAVE_PATH}")
        print("Training may have been interrupted or no model was saved.")
        return
    
    # Create evaluator
    evaluator = EPRNEvaluator(model, config.DEVICE)
    
    # Test overall accuracy
    try:
        test_acc, test_emo_acc, test_preds, test_labels = evaluator.evaluate(test_loader)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # Test per-emotion accuracy
    print("\n[6] Per-emotion analysis...")
    try:
        emotion_accs, emotion_samples, emotion_variance, emotion_mean, max_diff = evaluator.evaluate_per_emotion(
            test_loader, dataset, test_idx
        )
    except Exception as e:
        print(f"Error during per-emotion analysis: {e}")
        emotion_accs = []
        emotion_samples = []
        emotion_variance = 0
        emotion_mean = 0
        max_diff = 0
    
    # Calculate metrics
    metrics = calculate_metrics(
        checkpoint['train_acc'],
        checkpoint['val_acc'],
        test_acc,
        emotion_accs
    )
    
    # Prepare results dictionary
    results_dict = {
        'best_val_accuracy': checkpoint['val_acc'] * 100,
        'test_accuracy': test_acc * 100,
        'test_emotion_accuracy': test_emo_acc * 100,
        'emotion_variance': emotion_variance,
        'emotion_mean': emotion_mean,
        'max_emotion_diff': max_diff,
        'train_val_gap': metrics['train_val_gap'],
        'model_parameters': total_params,
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'final_epoch': checkpoint['epoch'] + 1,
        'per_emotion_accs': emotion_accs,
        'per_emotion_samples': emotion_samples
    }
    
    # Print results
    print_results_summary(results_dict)
    
    # Save results
    try:
        evaluator.save_results(results_dict, history)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # 7. Success evaluation
    print_success_message(test_acc, emotion_variance)
    
    # 8. Optional: Run visualizations if available
    try:
        from utils.visualization import EPRNVisualizer, plot_quick_results
        
        print("\n[7] Generating visualizations...")
        visualizer = EPRNVisualizer(config.RESULTS_FOLDER)
        
        # Plot training history
        visualizer.plot_training_history(history)
        
        # Plot quick results
        plot_quick_results(test_acc, checkpoint['val_acc'], emotion_accs)
        
        print("✓ Visualizations generated and saved to ./results/")
        
    except ImportError:
        print("\nNote: Visualization module not available or dependencies missing")
        print("Install matplotlib and seaborn for visualizations:")
        print("pip install matplotlib seaborn plotly")
    except Exception as e:
        print(f"\nNote: Could not generate visualizations: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()