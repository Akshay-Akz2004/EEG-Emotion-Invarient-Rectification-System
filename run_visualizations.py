# run_visualizations.py
import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append('.')

def run_visualizations():
    """Run all visualizations using existing results"""
    print("="*80)
    print("EPRN VISUALIZATION MODULE")
    print("="*80)
    
    # Check if results exist
    results_folder = './results'
    if not os.path.exists(results_folder):
        print(f"❌ Results folder not found: {results_folder}")
        print("Please run training first: python main.py")
        return
    
    # Check required files
    required_files = [
        'eprn_training_history.csv',
        'eprn_final_results.csv',
        'eprn_per_emotion_final.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(results_folder, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please run training first: python main.py")
        return
    
    print("✅ Found all required result files")
    
    # Load data
    try:
        history_df = pd.read_csv(os.path.join(results_folder, 'eprn_training_history.csv'))
        results_df = pd.read_csv(os.path.join(results_folder, 'eprn_final_results.csv'))
        per_emotion_df = pd.read_csv(os.path.join(results_folder, 'eprn_per_emotion_final.csv'))
        
        print("📊 Loaded data:")
        print(f"   - Training history: {len(history_df)} epochs")
        print(f"   - Test accuracy: {results_df['test_accuracy'].values[0]:.2f}%")
        print(f"   - Per-emotion results: {len(per_emotion_df)} emotions")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Import visualization module
    try:
        from utils.visualization import EPRNVisualizer, plot_quick_results
        
        print("\n🎨 Creating visualizations...")
        visualizer = EPRNVisualizer(results_folder)
        
        # 1. Convert history dataframe to dictionary
        history_dict = history_df.to_dict('list')
        
        # 2. Plot training history
        print("1. Plotting training history...")
        visualizer.plot_training_history(history_dict)
        
        # 3. Plot per-emotion performance
        print("2. Plotting per-emotion performance...")
        if 'accuracy' in per_emotion_df.columns:
            emotion_accs = per_emotion_df['accuracy'].values / 100
            emotion_names = per_emotion_df['emotion'].values
            visualizer.plot_per_emotion_performance(emotion_accs, emotion_names)
        
        # 4. Plot quick results
        print("3. Creating quick results summary...")
        test_acc = results_df['test_accuracy'].values[0] / 100
        val_acc = results_df['best_val_accuracy'].values[0] / 100
        
        if 'per_emotion_accs' in results_df.columns:
            # Parse string list if needed
            emotion_accs_str = results_df['per_emotion_accs'].values[0]
            if isinstance(emotion_accs_str, str):
                emotion_accs = eval(emotion_accs_str)
            else:
                emotion_accs = []
        else:
            emotion_accs = emotion_accs if 'emotion_accs' in locals() else []
        
        plot_quick_results(test_acc, val_acc, emotion_accs)
        
        # 5. Create emotion invariance plot
        print("4. Creating emotion invariance plot...")
        visualizer.plot_emotion_invariance(per_emotion_df)
        
        # 6. Create comprehensive report
        print("5. Generating HTML report...")
        
        results_dict = {
            'best_val_accuracy': results_df['best_val_accuracy'].values[0],
            'test_accuracy': results_df['test_accuracy'].values[0],
            'test_emotion_accuracy': results_df['test_emotion_accuracy'].values[0],
            'emotion_variance': results_df['emotion_variance'].values[0],
            'emotion_mean': results_df['emotion_mean'].values[0],
            'max_emotion_diff': results_df['max_emotion_diff'].values[0]
        }
        
        visualizer.create_summary_report(
            history=history_dict,
            test_results=results_dict,
            per_emotion_results=per_emotion_df.to_dict()
        )
        
        print("\n" + "="*80)
        print("✅ ALL VISUALIZATIONS COMPLETE!")
        print("="*80)
        print("\nGenerated files in ./results/:")
        print("  📈 training_history.png")
        print("  🎭 per_emotion_performance.png")
        print("  📊 quick_results.png")
        print("  📱 emotion_invariance.html/.png")
        print("  📄 summary_report.html")
        
        # Show generated files
        viz_files = [f for f in os.listdir(results_folder) 
                    if f.endswith(('.png', '.html', '.pdf'))]
        if viz_files:
            print(f"\nTotal visualization files: {len(viz_files)}")
            for file in sorted(viz_files):
                size = os.path.getsize(os.path.join(results_folder, file)) / 1024
                print(f"  - {file} ({size:.1f} KB)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nInstall required packages:")
        print("pip install matplotlib seaborn plotly")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_visualizations()