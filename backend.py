# auth_backend.py
import numpy as np
import torch
import os
import sys
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from user_database import UserDatabase

# Add project path
sys.path.append('.')

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = './results/best_eprn_robust_model.pth'
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global instances
model = None
db = UserDatabase()

def load_model():
    """Load the trained EPRN model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            print(f"📦 Loading model from {MODEL_PATH}")
            from config import config
            from model.eprn_model import RobustEPRN
            
            model = RobustEPRN(config.NUM_SUBJECTS, config.NUM_EMOTIONS)
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"✅ Model loaded successfully")
            print(f"   - Accuracy: {checkpoint['val_acc']*100:.2f}%")
            return True
        else:
            print("❌ Model not found at:", MODEL_PATH)
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

def extract_embedding_from_eeg(eeg_tensor):
    """Extract feature embedding from EEG using EPRN model"""
    if model is None:
        raise ValueError("Model not loaded")
    
    with torch.no_grad():
        # Get the rectified features from the model
        # We need to modify the model to return the rectified features
        subject_logits, emotion_logits, rectified_features, _ = model(
            eeg_tensor, return_components=True
        )
        
        # Use the rectified features as embedding (these are emotion-invariant)
        embedding = rectified_features.mean(dim=0).numpy()  # Average across time
        
        # Also get predictions for display
        subject_pred = torch.argmax(subject_logits, dim=1).item()
        emotion_pred = torch.argmax(emotion_logits, dim=1).item()
        subject_conf = torch.softmax(subject_logits, dim=1)[0, subject_pred].item()
        emotion_conf = torch.softmax(emotion_logits, dim=1)[0, emotion_pred].item()
        
        return {
            'embedding': embedding.tolist(),
            'subject_id': subject_pred + 1,
            'emotion': ['Happy', 'Sad', 'Fear', 'Disgust'][emotion_pred],
            'subject_confidence': subject_conf * 100,
            'emotion_confidence': emotion_conf * 100
        }

def process_eeg_file(file_path):
    """Process uploaded EEG file"""
    try:
        data = np.load(file_path)
        
        # Find the features array
        if 'features' in data:
            features = data['features']
        elif 'eeg' in data:
            features = data['eeg']
        elif 'data' in data:
            features = data['data']
        else:
            first_key = list(data.keys())[0]
            features = data[first_key]
        
        # Ensure correct shape
        if len(features.shape) == 3:
            # Already in (samples, time, channels) format
            pass
        elif len(features.shape) == 2:
            # Try to reshape
            if features.shape[1] == 330:  # 66 * 5
                features = features.reshape(-1, 66, 5)
            else:
                raise ValueError(f"Cannot reshape features with shape {features.shape}")
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        # Take first sample
        sample = features[0:1]
        
        # Normalize
        sample = (sample - sample.mean()) / (sample.std() + 1e-8)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(sample)  # Shape: [1, 66, 5]
        
        return tensor_data, None
        
    except Exception as e:
        return None, f"File processing error: {str(e)}"

# Load model at startup
model_loaded = load_model()

@app.route('/')
def home():
    return jsonify({
        'service': 'EEG Biometric Authentication System',
        'model_loaded': model_loaded,
        'registered_users': len(db.list_users()),
        'authentication_threshold': '70%',
        'endpoints': {
            '/register': 'POST - Register new user (2 EEG files)',
            '/authenticate': 'POST - Authenticate user (1 EEG file)',
            '/users': 'GET - List registered users',
            '/user/<id>': 'GET - Get user profile',
            '/delete/<id>': 'DELETE - Delete user',
            '/reset': 'POST - Reset database'
        }
    })

@app.route('/register', methods=['POST'])
def register_user():
    """Register a new user with 2 EEG files"""
    try:
        # Get form data
        user_id = request.form.get('user_id')
        username = request.form.get('username')
        email = request.form.get('email')
        
        if not all([user_id, username]):
            return jsonify({'error': 'Missing user information'}), 400
        
        # Check for EEG files
        if 'eeg_file1' not in request.files or 'eeg_file2' not in request.files:
            return jsonify({'error': 'Please upload 2 EEG files'}), 400
        
        file1 = request.files['eeg_file1']
        file2 = request.files['eeg_file2']
        
        if not (file1.filename.endswith('.npz') and file2.filename.endswith('.npz')):
            return jsonify({'error': 'Only .npz files accepted'}), 400
        
        # Process both files
        embeddings = []
        results = []
        
        for i, file in enumerate([file1, file2], 1):
            # Save file
            temp_path = os.path.join(UPLOAD_FOLDER, f'temp_{user_id}_{i}.npz')
            file.save(temp_path)
            
            # Process file
            eeg_tensor, error = process_eeg_file(temp_path)
            if error:
                os.remove(temp_path)
                return jsonify({'error': f'File {i}: {error}'}), 400
            
            # Extract embedding
            if model_loaded:
                result = extract_embedding_from_eeg(eeg_tensor)
                embeddings.append(result['embedding'])
                results.append({
                    'file': i,
                    'subject_id': result['subject_id'],
                    'emotion': result['emotion'],
                    'subject_confidence': round(result['subject_confidence'], 2),
                    'emotion_confidence': round(result['emotion_confidence'], 2)
                })
            else:
                # Simulation mode
                embeddings.append(np.random.randn(128).tolist())
                results.append({
                    'file': i,
                    'subject_id': np.random.randint(1, 17),
                    'emotion': np.random.choice(['Happy', 'Sad', 'Fear', 'Disgust']),
                    'subject_confidence': round(80 + np.random.rand() * 20, 2),
                    'emotion_confidence': round(75 + np.random.rand() * 25, 2)
                })
            
            # Clean up
            os.remove(temp_path)
        
        # Register user in database
        success, message = db.register_user(user_id, username, email, embeddings)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'user_id': user_id,
                'username': username,
                'embedding_dim': len(embeddings[0]),
                'file_results': results
            })
        else:
            return jsonify({'error': message}), 400
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Authenticate user with 1 EEG file"""
    try:
        user_id = request.form.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if 'eeg_file' not in request.files:
            return jsonify({'error': 'No EEG file provided'}), 400
        
        file = request.files['eeg_file']
        if not file.filename.endswith('.npz'):
            return jsonify({'error': 'Only .npz files accepted'}), 400
        
        # Save file
        temp_path = os.path.join(UPLOAD_FOLDER, f'auth_{user_id}.npz')
        file.save(temp_path)
        
        # Process file
        eeg_tensor, error = process_eeg_file(temp_path)
        if error:
            os.remove(temp_path)
            return jsonify({'error': error}), 400
        
        # Extract embedding from authentication file
        if model_loaded:
            result = extract_embedding_from_eeg(eeg_tensor)
            new_embedding = result['embedding']
            auth_result = result
        else:
            # Simulation mode
            new_embedding = np.random.randn(128).tolist()
            auth_result = {
                'subject_id': np.random.randint(1, 17),
                'emotion': np.random.choice(['Happy', 'Sad', 'Fear', 'Disgust']),
                'subject_confidence': round(80 + np.random.rand() * 20, 2),
                'emotion_confidence': round(75 + np.random.rand() * 25, 2)
            }
        
        # Authenticate against stored profile
        threshold = float(request.form.get('threshold', 0.7))
        is_authentic, similarity, used_threshold = db.authenticate_user(
            user_id, new_embedding, threshold
        )
        
        # Get user info
        user = db.get_user(user_id)
        
        response = {
            'authenticated': bool(is_authentic),
            'similarity_score': round(similarity * 100, 2),
            'threshold_used': round(used_threshold * 100, 2),
            'user_id': user_id,
            'username': user['username'] if user else 'Unknown',
            'registration_date': user['registration_date'] if user else None,
            'auth_attempts': user['auth_attempts'] if user else 0,
            'successful_auths': user['successful_auths'] if user else 0,
            'current_auth_result': auth_result,
            'decision': 'ACCESS GRANTED' if is_authentic else 'ACCESS DENIED'
        }
        
        os.remove(temp_path)
        return jsonify(response)
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/users', methods=['GET'])
def list_users():
    """List all registered users"""
    users = db.list_users()
    user_details = []
    
    for user_id in users:
        user = db.get_user(user_id)
        if user:
            user_details.append({
                'user_id': user_id,
                'username': user['username'],
                'email': user['email'],
                'registration_date': user['registration_date'],
                'auth_attempts': user['auth_attempts'],
                'successful_auths': user['successful_auths']
            })
    
    return jsonify({
        'count': len(users),
        'users': user_details
    })

@app.route('/user/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get user profile"""
    user = db.get_user(user_id)
    if user:
        return jsonify(user)
    else:
        return jsonify({'error': 'User not found'}), 404

@app.route('/delete/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete user"""
    if db.delete_user(user_id):
        return jsonify({'success': True, 'message': f'User {user_id} deleted'})
    else:
        return jsonify({'error': 'User not found'}), 404

@app.route('/reset', methods=['POST'])
def reset_db():
    """Reset database"""
    db.reset_database()
    return jsonify({'success': True, 'message': 'Database reset'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔐 EEG BIOMETRIC AUTHENTICATION SYSTEM")
    print("="*70)
    print(f"📊 Model loaded: {model_loaded}")
    print(f"👥 Registered users: {len(db.list_users())}")
    print(f"📁 Database: {db.db_path}")
    print(f"🌐 API: http://localhost:5000")
    print("\n📋 API Endpoints:")
    print("  POST /register     - Register new user (2 EEG files)")
    print("  POST /authenticate - Authenticate user (1 EEG file)")
    print("  GET  /users        - List all users")
    print("  GET  /user/<id>    - Get user details")
    print("  DELETE /delete/<id>- Delete user")
    print("  POST /reset        - Reset database")
    print("="*70)
    app.run(host='0.0.0.0', port=5000, debug=True)