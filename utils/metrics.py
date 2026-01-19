# utils/metrics.py
import numpy as np

def calculate_metrics(train_acc, val_acc, test_acc, emotion_accs):
    """Calculate various metrics"""
    # Overfitting gap
    train_val_gap = (train_acc - val_acc) * 100
    
    # Emotion invariance metrics
    if emotion_accs:
        emotion_variance = np.std(emotion_accs) * 100
        emotion_mean = np.mean(emotion_accs) * 100
        max_emotion_diff = (max(emotion_accs) - min(emotion_accs)) * 100
    else:
        emotion_variance = emotion_mean = max_emotion_diff = 0
    
    metrics = {
        'train_val_gap': train_val_gap,
        'emotion_variance': emotion_variance,
        'emotion_mean': emotion_mean,
        'max_emotion_diff': max_emotion_diff,
        'test_accuracy': test_acc * 100
    }
    
    return metrics

def print_results_summary(results_dict):
    """Print formatted results summary"""
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if 'best_val_accuracy' in results_dict:
        print(f"Best Validation Accuracy: {results_dict['best_val_accuracy']:.2f}%")
    if 'test_accuracy' in results_dict:
        print(f"Test Accuracy:           {results_dict['test_accuracy']:.2f}%")
    if 'test_emotion_accuracy' in results_dict:
        print(f"Test Emotion Accuracy:   {results_dict['test_emotion_accuracy']:.2f}%")
    
    if 'emotion_variance' in results_dict and results_dict['emotion_variance'] > 0:
        print(f"\nEmotion-Invariance Metrics:")
        print(f"  Mean per-emotion:      {results_dict['emotion_mean']:.2f}%")
        print(f"  Standard Deviation:    {results_dict['emotion_variance']:.2f}%")
        print(f"  Max Difference:        {results_dict['max_emotion_diff']:.2f}%")
    
    print("="*80)
    
    if 'train_val_gap' in results_dict:
        print(f"\nOverfitting Analysis:")
        print(f"  Final Train-Val Gap:   {results_dict['train_val_gap']:.1f}%")
        if results_dict['train_val_gap'] > 5:
            print("  ⚠ Warning: Potential overfitting detected")
        else:
            print("  ✓ Good: Minimal overfitting")