# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os

# Import plotly only when needed to avoid circular imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from sklearn.metrics import confusion_matrix, classification_report

class EPRNVisualizer:
    """Visualization utilities for EPRN model analysis"""
    
    def __init__(self, results_folder='./results'):
        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = sns.color_palette("husl", 8)
        self.emotion_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Red, Teal, Blue, Green
        
    def plot_training_history(self, history, save_path=None):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Accuracy plots
        axes[0, 0].plot(history['train_acc'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_acc'], label='Val', linewidth=2)
        axes[0, 0].set_title('Subject Classification Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plots
        axes[0, 1].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Emotion accuracy
        axes[0, 2].plot(history['train_emo_acc'], label='Train', linewidth=2)
        axes[0, 2].plot(history['val_emo_acc'], label='Val', linewidth=2)
        axes[0, 2].set_title('Emotion Classification Accuracy', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Accuracy gap (overfitting indicator)
        train_val_gap = [(train - val) * 100 for train, val in zip(history['train_acc'], history['val_acc'])]
        axes[1, 0].plot(train_val_gap, color='red', linewidth=2)
        axes[1, 0].axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Warning threshold (5%)')
        axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[1, 0].fill_between(range(len(train_val_gap)), 0, train_val_gap, 
                               alpha=0.3, color='red')
        axes[1, 0].set_title('Train-Val Accuracy Gap (Overfitting)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gap (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(history['lr'], color='purple', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined metric
        axes[1, 2].plot([acc * 100 for acc in history['val_acc']], label='Val Acc', linewidth=2)
        axes[1, 2].plot([acc * 100 for acc in history['val_emo_acc']], label='Val Emotion Acc', linewidth=2)
        axes[1, 2].set_title('Validation Performance', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy (%)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('EPRN Training History', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_folder, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title='Confusion Matrix', 
                             normalize=True, save_path=None):
        """Plot confusion matrix"""
        if labels is None:
            labels = [f'Subj {i+1}' for i in range(len(np.unique(y_true)))]
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            vmin, vmax = 0, 1
        else:
            fmt = 'd'
            vmin, vmax = None, None
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   vmin=vmin, vmax=vmax,
                   cbar_kws={'label': 'Normalized Accuracy' if normalize else 'Count'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path is None:
            suffix = '_normalized' if normalize else ''
            save_path = os.path.join(self.results_folder, f'confusion_matrix{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_per_emotion_performance(self, emotion_accs, emotion_names=None, save_path=None):
        """Plot performance for each emotion category"""
        if emotion_names is None:
            emotion_names = ['Happy', 'Sad', 'Fear', 'Disgust']
        
        plt.figure(figsize=(10, 6))
        
        # Create bars
        bars = plt.bar(range(len(emotion_accs)), [acc * 100 for acc in emotion_accs], 
                      color=self.emotion_colors[:len(emotion_accs)], edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, emotion_accs)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Calculate and plot statistics
        mean_acc = np.mean(emotion_accs) * 100
        std_acc = np.std(emotion_accs) * 100
        
        plt.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_acc:.1f}%')
        plt.fill_between([-0.5, len(emotion_accs)-0.5], 
                        [mean_acc - std_acc, mean_acc - std_acc],
                        [mean_acc + std_acc, mean_acc + std_acc],
                        alpha=0.2, color='red', label=f'Std: ±{std_acc:.1f}%')
        
        plt.title('Authentication Accuracy Per Emotion', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion Category', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xticks(range(len(emotion_accs)), emotion_names)
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add variance annotation
        plt.text(0.02, 0.98, f'Variance: {std_acc:.2f}%', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path is None:
            save_path = os.path.join(self.results_folder, 'per_emotion_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def plot_feature_importance(self, model, sample_input, save_path=None):
        """Visualize feature importance using gradient-based analysis"""
        model.eval()
        
        # Ensure sample_input requires gradient
        sample_input = sample_input.clone().detach().requires_grad_(True)
        
        # Forward pass
        subject_logits, emotion_logits = model(sample_input.unsqueeze(0))
        
        # Compute gradients w.r.t input
        subject_logits[0, sample_input.shape[0] % 16].backward()  # Use a target class
        
        # Get gradients
        gradients = sample_input.grad.abs().cpu().numpy()
        
        # Average across channels and time
        channel_importance = gradients.mean(axis=(0, 1))  # Average across time and batch
        time_importance = gradients.mean(axis=(0, 2))     # Average across channels and batch
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Channel importance
        channels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        axes[0].bar(channels, channel_importance, color=self.colors[:len(channels)])
        axes[0].set_title('EEG Band Importance', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Frequency Band')
        axes[0].set_ylabel('Gradient Magnitude')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Time importance (simplified)
        axes[1].plot(time_importance, linewidth=2, color=self.colors[5])
        axes[1].set_title('Temporal Importance', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Gradient Magnitude')
        axes[1].grid(True, alpha=0.3)
        
        # Heatmap of gradients
        im = axes[2].imshow(gradients[0].T, aspect='auto', cmap='viridis')
        axes[2].set_title('Input Gradient Heatmap', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('EEG Band')
        axes[2].set_yticks(range(len(channels)))
        axes[2].set_yticklabels(channels)
        plt.colorbar(im, ax=axes[2], label='Gradient Magnitude')
        
        plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_folder, 'feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return gradients
    
    def plot_emotion_invariance(self, per_emotion_results, save_path=None):
        """Plot emotion invariance analysis"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Using matplotlib instead.")
            return self._plot_emotion_invariance_matplotlib(per_emotion_results, save_path)
        
        emotion_names = per_emotion_results['emotion']
        accuracies = per_emotion_results['accuracy']
        
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=emotion_names,
            y=accuracies,
            marker_color=self.emotion_colors[:len(emotion_names)],
            marker_line_color='black',
            marker_line_width=1.5,
            text=[f'{acc:.1f}%' for acc in accuracies],
            textposition='outside',
            name='Accuracy'
        ))
        
        # Add mean line
        mean_acc = np.mean(accuracies)
        fig.add_hline(y=mean_acc, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_acc:.1f}%",
                     annotation_position="bottom right")
        
        # Add standard deviation band
        std_acc = np.std(accuracies)
        fig.add_hrect(y0=mean_acc - std_acc, y1=mean_acc + std_acc,
                     line_width=0, fillcolor="red", opacity=0.2,
                     annotation_text=f"±{std_acc:.1f}%",
                     annotation_position="top right")
        
        fig.update_layout(
            title={
                'text': "Emotion-Invariant Authentication Performance",
                'font': {'size': 20, 'family': "Arial, sans-serif"}
            },
            xaxis_title="Emotion Category",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 105],
            showlegend=False,
            template="plotly_white",
            width=800,
            height=500
        )
        
        if save_path is None:
            save_path = os.path.join(self.results_folder, 'emotion_invariance.html')
        fig.write_html(save_path)
        
        # Also save as PNG if kaleido is available
        try:
            png_path = save_path.replace('.html', '.png')
            fig.write_image(png_path, width=800, height=500)
        except:
            pass
        
        return fig
    
    def _plot_emotion_invariance_matplotlib(self, per_emotion_results, save_path=None):
        """Fallback matplotlib version for emotion invariance plot"""
        emotion_names = per_emotion_results['emotion']
        accuracies = per_emotion_results['accuracy']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(emotion_names, accuracies, color=self.emotion_colors[:len(emotion_names)])
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2., acc + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add mean and std
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        plt.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_acc:.1f}%')
        plt.fill_between([-0.5, len(emotion_names)-0.5],
                        [mean_acc - std_acc, mean_acc - std_acc],
                        [mean_acc + std_acc, mean_acc + std_acc],
                        alpha=0.2, color='red', label=f'Std: ±{std_acc:.1f}%')
        
        plt.title('Emotion-Invariant Authentication Performance', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion Category')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = os.path.join(self.results_folder, 'emotion_invariance_matplotlib.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def create_summary_report(self, history, test_results, per_emotion_results, 
                             confusion_matrix_data=None, save_path=None):
        """Create a comprehensive summary report with all visualizations"""
        if save_path is None:
            save_path = os.path.join(self.results_folder, 'summary_report.html')
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EPRN Model Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; min-width: 200px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .good {{ color: #28a745; font-weight: bold; }}
                .warning {{ color: #ffc107; font-weight: bold; }}
                .bad {{ color: #dc3545; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>EPRN Model Summary Report</h1>
                <p>EEG Biometric Authentication with Emotion-Invariant Features</p>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Overall Performance</h2>
                <div class="metric">
                    <div class="metric-value">{test_results.get('test_accuracy', 0):.2f}%</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{test_results.get('best_val_accuracy', 0):.2f}%</div>
                    <div class="metric-label">Best Validation Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{test_results.get('test_emotion_accuracy', 0):.2f}%</div>
                    <div class="metric-label">Emotion Recognition Accuracy</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Emotion Invariance Analysis</h2>
        """
        
        # Add emotion invariance metrics
        if per_emotion_results is not None:
            mean_acc = per_emotion_results.get('emotion_mean', 0)
            variance = per_emotion_results.get('emotion_variance', 0)
            
            variance_class = "good" if variance < 2 else "warning" if variance < 5 else "bad"
            
            html_content += f"""
                <div class="metric">
                    <div class="metric-value">{mean_acc:.2f}%</div>
                    <div class="metric-label">Mean Per-Emotion Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value {variance_class}">{variance:.2f}%</div>
                    <div class="metric-label">Emotion Variance</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{per_emotion_results.get('max_emotion_diff', 0):.2f}%</div>
                    <div class="metric-label">Max Emotion Difference</div>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Training Statistics</h2>
                <div class="image-container">
                    <img src="training_history.png" alt="Training History">
                </div>
            </div>
            
            <div class="section">
                <h2>Per-Emotion Performance</h2>
                <div class="image-container">
                    <img src="per_emotion_performance.png" alt="Per-Emotion Performance">
                </div>
            </div>
        """
        
        if confusion_matrix_data is not None:
            html_content += """
            <div class="section">
                <h2>Confusion Matrix</h2>
                <div class="image-container">
                    <img src="confusion_matrix_normalized.png" alt="Confusion Matrix">
                </div>
            </div>
            """
        
        html_content += """
            <div class="section">
                <h2>Overfitting Analysis</h2>
        """
        
        # Add overfitting analysis
        if history and 'train_acc' in history and 'val_acc' in history:
            final_gap = (history['train_acc'][-1] - history['val_acc'][-1]) * 100
            gap_class = "good" if final_gap < 5 else "warning" if final_gap < 10 else "bad"
            
            html_content += f"""
                <div class="metric">
                    <div class="metric-value {gap_class}">{final_gap:.1f}%</div>
                    <div class="metric-label">Final Train-Val Gap</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(history['train_acc'])}</div>
                    <div class="metric-label">Training Epochs</div>
                </div>
                <p><strong>Interpretation:</strong> {self._get_overfitting_interpretation(final_gap)}</p>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Add recommendations based on results
        test_acc = test_results.get('test_accuracy', 0)
        if test_acc >= 85:
            html_content += """
                    <li>✓ Model performance is excellent! (>85% accuracy)</li>
                    <li>✓ Emotion invariance looks good</li>
                    <li>✓ Ready for publication/presentation</li>
            """
        elif test_acc >= 75:
            html_content += """
                    <li>⚠ Model performance is good but could be improved</li>
                    <li>✓ Consider increasing training epochs</li>
                    <li>✓ Try different learning rate schedules</li>
            """
        else:
            html_content += """
                    <li>⚠ Model needs improvement</li>
                    <li>✓ Increase model capacity</li>
                    <li>✓ Add more regularization</li>
                    <li>✓ Consider data augmentation</li>
            """
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Summary report saved to {save_path}")
        
        return html_content
    
    def _get_overfitting_interpretation(self, gap):
        """Get interpretation of overfitting gap"""
        if gap < 2:
            return "Excellent: Minimal overfitting detected."
        elif gap < 5:
            return "Good: Acceptable level of overfitting."
        elif gap < 10:
            return "Moderate: Some overfitting present. Consider adding regularization."
        else:
            return "High: Significant overfitting. Add dropout, weight decay, or get more data."

# Quick visualization functions (standalone)
def plot_quick_results(test_acc, val_acc, emotion_accs, save_path=None):
    """Quick plot of essential results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall accuracy
    categories = ['Validation', 'Test']
    accuracies = [val_acc * 100, test_acc * 100]
    
    bars1 = ax1.bar(categories, accuracies, color=['#4ECDC4', '#FF6B6B'])
    ax1.set_title('Overall Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Per-emotion accuracy
    if emotion_accs:
        emotion_names = ['Happy', 'Sad', 'Fear', 'Disgust'][:len(emotion_accs)]
        bars2 = ax2.bar(emotion_names, [acc * 100 for acc in emotion_accs], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(emotion_accs)])
        ax2.set_title('Per-Emotion Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add variance info
        variance = np.std(emotion_accs) * 100
        ax2.text(0.02, 0.98, f'Variance: {variance:.2f}%', 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('EPRN Authentication Results', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    if save_path is None:
        save_path = './results/quick_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_eeg_sample(sample_data, title="EEG Sample", save_path=None):
    """Visualize a sample EEG data"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 8))
    
    # Plot each channel
    channels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    time_points = range(sample_data.shape[0])
    
    for i in range(sample_data.shape[1]):
        plt.plot(time_points, sample_data[:, i], label=channels[i], linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time (samples)', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path is None:
        save_path = './results/eeg_sample.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()