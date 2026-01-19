# user_database.py
import json
import numpy as np
import os
import pickle
from datetime import datetime

class UserDatabase:
    """Database for storing user EEG profiles"""
    
    def __init__(self, db_path='./user_profiles.pkl'):
        self.db_path = db_path
        self.users = {}
        self.load()
    
    def load(self):
        """Load database from file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.users = pickle.load(f)
                print(f"✅ Loaded {len(self.users)} users from database")
            except:
                self.users = {}
                print("⚠ Created new database (file corrupted)")
        else:
            self.users = {}
            print("📁 Created new user database")
    
    def save(self):
        """Save database to file"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.users, f)
        print(f"💾 Database saved: {len(self.users)} users")
    
    def register_user(self, user_id, username, email, eeg_embeddings):
        """Register a new user with their EEG embeddings"""
        if user_id in self.users:
            return False, "User already exists"
        
        # Create user profile
        profile = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'embeddings': eeg_embeddings,  # List of feature embeddings
            'signature': self._create_signature(eeg_embeddings),
            'registration_date': datetime.now().isoformat(),
            'auth_attempts': 0,
            'successful_auths': 0
        }
        
        self.users[user_id] = profile
        self.save()
        return True, "User registered successfully"
    
    def _create_signature(self, embeddings):
        """Create a unique signature from multiple embeddings"""
        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        # Normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        return avg_embedding.tolist()
    
    def authenticate_user(self, user_id, new_embedding, threshold=0.7):
        """Authenticate user by comparing new embedding with stored signature"""
        if user_id not in self.users:
            return False, "User not found", 0.0
        
        user = self.users[user_id]
        
        # Update auth attempts
        user['auth_attempts'] += 1
        
        # Convert to numpy arrays
        stored_signature = np.array(user['signature'])
        new_embedding_np = np.array(new_embedding)
        
        # Normalize new embedding
        norm = np.linalg.norm(new_embedding_np)
        if norm > 0:
            new_embedding_np = new_embedding_np / norm
        
        # Calculate cosine similarity
        similarity = np.dot(stored_signature, new_embedding_np)
        
        # Check against threshold
        is_authentic = similarity >= threshold
        
        if is_authentic:
            user['successful_auths'] += 1
        
        # Update database
        self.save()
        
        return is_authentic, similarity, threshold
    
    def get_user(self, user_id):
        """Get user profile"""
        return self.users.get(user_id)
    
    def list_users(self):
        """List all registered users"""
        return list(self.users.keys())
    
    def delete_user(self, user_id):
        """Delete user from database"""
        if user_id in self.users:
            del self.users[user_id]
            self.save()
            return True
        return False
    
    def reset_database(self):
        """Reset entire database"""
        self.users = {}
        self.save()
        return True