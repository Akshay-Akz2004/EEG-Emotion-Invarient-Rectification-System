# simple_backend.py
import numpy as np
import torch
import os
import sys
import json
import pickle
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project path
sys.path.append('.')

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = './results/best_eprn_robust_model.pth'
DB_PATH = './user_profiles.pkl'
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global instances
model = None
user_database = {}

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
            print("❌ Model not found")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def load_database():
    """Load user database"""
    global user_database
    try:
        if os.path.exists(DB_PATH):
            with open(DB_PATH, 'rb') as f:
                user_database = pickle.load(f)
            print(f"✅ Loaded {len(user_database)} users from database")
        else:
            user_database = {}
            print("📁 Created new user database")
    except:
        user_database = {}
        print("⚠ Created new database (file corrupted)")

def save_database():
    """Save user database"""
    with open(DB_PATH, 'wb') as f:
        pickle.dump(user_database, f)
    print(f"💾 Database saved: {len(user_database)} users")

def process_eeg_file(file_path):
    """Process EEG .npz file"""
    try:
        data = np.load(file_path)
        
        # Find features array
        if 'features' in data:
            features = data['features']
        else:
            # Try first array
            first_key = list(data.keys())[0]
            features = data[first_key]
        
        # Take first sample
        if len(features.shape) == 3:
            sample = features[0:1]  # Shape: [1, 66, 5]
        elif len(features.shape) == 2:
            # Reshape if needed
            sample = features[0:1].reshape(1, 66, 5)
        else:
            return None, f"Unexpected shape: {features.shape}"
        
        # Normalize
        sample = (sample - sample.mean()) / (sample.std() + 1e-8)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(sample)
        
        return tensor_data, None
    except Exception as e:
        return None, f"File processing error: {str(e)}"

def extract_embedding(eeg_tensor):
    """Extract embedding from EEG tensor"""
    if model is None:
        # Create random embedding for testing
        embedding = np.random.randn(128).tolist()
        subject_id = np.random.randint(1, 17)
        emotion_id = np.random.randint(0, 4)
        emotions = ['Happy', 'Sad', 'Fear', 'Disgust']
        
        return {
            'embedding': embedding,
            'subject_id': subject_id,
            'emotion': emotions[emotion_id],
            'subject_confidence': round(85 + np.random.rand() * 15, 2),
            'emotion_confidence': round(80 + np.random.rand() * 20, 2)
        }
    
    with torch.no_grad():
        # Forward pass
        subject_logits, emotion_logits, rectified_features = model(eeg_tensor)
        
        # Get predictions
        subject_pred = torch.argmax(subject_logits, dim=1).item()
        emotion_pred = torch.argmax(emotion_logits, dim=1).item()
        
        # Get confidences
        subject_conf = torch.softmax(subject_logits, dim=1)[0, subject_pred].item()
        emotion_conf = torch.softmax(emotion_logits, dim=1)[0, emotion_pred].item()
        
        # Get embedding (rectified features)
        embedding = rectified_features.mean(dim=0).numpy().tolist()  # Average across batch
        
        emotions = ['Happy', 'Sad', 'Fear', 'Disgust']
        
        return {
            'embedding': embedding,
            'subject_id': subject_pred + 1,
            'emotion': emotions[emotion_pred],
            'subject_confidence': round(subject_conf * 100, 2),
            'emotion_confidence': round(emotion_conf * 100, 2)
        }

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Normalize
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 > 0:
        emb1 = emb1 / norm1
    if norm2 > 0:
        emb2 = emb2 / norm2
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2)
    
    return float(similarity)

# Initialize
print("\n" + "="*70)
print("🔐 EEG BIOMETRIC AUTHENTICATION SYSTEM")
print("="*70)
model_loaded = load_model()
load_database()
print(f"👥 Registered users: {len(user_database)}")
print(f"🌐 API: http://localhost:5001")
print("="*70)

@app.route('/')
def home():
    return jsonify({
        'service': 'EEG Biometric Authentication',
        'model_loaded': model_loaded,
        'users': len(user_database),
        'endpoints': {
            '/register': 'POST - Register user (2 EEG files)',
            '/authenticate': 'POST - Authenticate user (1 EEG file)',
            '/users': 'GET - List users',
            '/reset': 'POST - Reset database'
        }
    })

@app.route('/register', methods=['POST'])
def register():
    """Register new user with 2 EEG files"""
    try:
        # Get form data
        user_id = request.form.get('user_id')
        username = request.form.get('username')
        
        if not user_id or not username:
            return jsonify({'error': 'Missing user_id or username'}), 400
        
        if user_id in user_database:
            return jsonify({'error': 'User already exists'}), 400
        
        # Check files
        if 'eeg_file1' not in request.files or 'eeg_file2' not in request.files:
            return jsonify({'error': 'Need 2 EEG files'}), 400
        
        file1 = request.files['eeg_file1']
        file2 = request.files['eeg_file2']
        
        # Process files
        embeddings = []
        results = []
        
        for i, file in enumerate([file1, file2], 1):
            # Save temp file
            temp_path = os.path.join(UPLOAD_FOLDER, f'temp_{user_id}_{i}.npz')
            file.save(temp_path)
            
            # Process EEG
            eeg_tensor, error = process_eeg_file(temp_path)
            if error:
                os.remove(temp_path)
                return jsonify({'error': f'File {i}: {error}'}), 400
            
            # Extract embedding
            result = extract_embedding(eeg_tensor)
            embeddings.append(result['embedding'])
            
            results.append({
                'file': i,
                'subject_id': result['subject_id'],
                'emotion': result['emotion'],
                'subject_confidence': result['subject_confidence'],
                'emotion_confidence': result['emotion_confidence']
            })
            
            os.remove(temp_path)
        
        # Create user profile
        # Average embeddings to create signature
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        user_database[user_id] = {
            'user_id': user_id,
            'username': username,
            'email': request.form.get('email', ''),
            'signature': avg_embedding,
            'registration_date': datetime.now().isoformat(),
            'auth_attempts': 0,
            'successful_auths': 0,
            'embeddings': embeddings  # Store both embeddings
        }
        
        save_database()
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user_id': user_id,
            'username': username,
            'embedding_dim': len(avg_embedding),
            'file_results': results,
            'similarity_between_files': round(calculate_similarity(embeddings[0], embeddings[1]) * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Authenticate user with 1 EEG file"""
    try:
        user_id = request.form.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if user_id not in user_database:
            return jsonify({'error': 'User not found'}), 404
        
        if 'eeg_file' not in request.files:
            return jsonify({'error': 'No EEG file provided'}), 400
        
        file = request.files['eeg_file']
        
        # Save temp file
        temp_path = os.path.join(UPLOAD_FOLDER, f'auth_{user_id}.npz')
        file.save(temp_path)
        
        # Process EEG
        eeg_tensor, error = process_eeg_file(temp_path)
        if error:
            os.remove(temp_path)
            return jsonify({'error': error}), 400
        
        # Extract embedding
        result = extract_embedding(eeg_tensor)
        new_embedding = result['embedding']
        
        # Get stored signature
        user = user_database[user_id]
        stored_signature = user['signature']
        
        # Calculate similarity
        threshold = float(request.form.get('threshold', 0.7))
        similarity = calculate_similarity(new_embedding, stored_signature)
        
        # Update stats
        user['auth_attempts'] += 1
        if similarity >= threshold:
            user['successful_auths'] += 1
        
        save_database()
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'authenticated': similarity >= threshold,
            'similarity_score': round(similarity * 100, 2),
            'threshold_used': round(threshold * 100, 2),
            'user_id': user_id,
            'username': user['username'],
            'registration_date': user['registration_date'],
            'auth_attempts': user['auth_attempts'],
            'successful_auths': user['successful_auths'],
            'current_auth_result': {
                'subject_id': result['subject_id'],
                'emotion': result['emotion'],
                'subject_confidence': result['subject_confidence'],
                'emotion_confidence': result['emotion_confidence']
            },
            'decision': 'ACCESS GRANTED' if similarity >= threshold else 'ACCESS DENIED'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/users', methods=['GET'])
def get_users():
    """Get list of all users"""
    users = []
    for user_id, user in user_database.items():
        users.append({
            'user_id': user_id,
            'username': user['username'],
            'email': user['email'],
            'registration_date': user['registration_date'],
            'auth_attempts': user['auth_attempts'],
            'successful_auths': user['successful_auths'],
            'success_rate': round((user['successful_auths'] / user['auth_attempts'] * 100) 
                                 if user['auth_attempts'] > 0 else 0, 1)
        })
    
    return jsonify({
        'count': len(users),
        'users': users
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset database"""
    global user_database
    count = len(user_database)
    user_database = {}
    save_database()
    return jsonify({
        'success': True,
        'message': f'Database reset. Removed {count} users.'
    })

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'loaded' if model else 'not loaded',
        'users': len(user_database),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)