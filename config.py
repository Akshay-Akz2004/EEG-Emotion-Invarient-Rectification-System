# config.py
import torch

class Config:
    """Configuration class for EPRN model"""
    # General
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    DATA_FOLDER = './processed_features'
    NUM_SUBJECTS = 16
    NUM_EMOTIONS = 4
    SAMPLE_PER_FILE = 5000
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Model
    INPUT_CHANNELS = 5
    INPUT_LENGTH = 66
    ENCODER_CHANNELS = [32, 64, 128]
    DROPOUT_RATE = 0.4
    SUBJECT_HEAD_HIDDEN = 256
    EMOTION_HEAD_HIDDEN = 64
    RECTIFICATION_ALPHA = 0.8
    
    # Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 40
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999
    LABEL_SMOOTHING = 0.15
    LAMBDA_EMOTION = 0.2
    GRADIENT_CLIP = 1.0
    
    # Scheduler
    STEP_SIZE = 10
    GAMMA = 0.5
    
    # Early Stopping
    PATIENCE_LIMIT = 15
    
    # Paths
    RESULTS_FOLDER = './results'
    MODEL_SAVE_PATH = './results/best_eprn_robust_model.pth'
    RESULTS_CSV = './results/eprn_final_results.csv'
    PER_EMOTION_CSV = './results/eprn_per_emotion_final.csv'
    HISTORY_CSV = './results/eprn_training_history.csv'
    
    # Emotion names for visualization
    EMOTION_NAMES = ['Happy', 'Sad', 'Fear', 'Disgust']
    
config = Config()