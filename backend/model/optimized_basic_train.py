"""
Optimized Basic Emotion Detection - Focus on core emotions with maximum processing power
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from preprocess import TextPreprocessor
import re

class OptimizedBasicEmotionTrainer:
    """Optimized trainer for basic emotions with maximum processing power"""
    
    def __init__(self, dataset_path="../../dataset/comprehensive_emotions.csv"):
        self.dataset_path = dataset_path
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Focus on basic, well-known emotions
        self.basic_emotions = {
            'happy': ['happy', 'joy', 'excitement', 'delight', 'elated', 'jubilant', 'ecstatic', 'euphoric', 'euphoria', 'blissful', 'bliss', 'overjoyed', 'thrilled', 'contentment'],
            'sad': ['sad', 'sorrow', 'melancholy', 'disappointment', 'grief', 'depressed', 'melancholic', 'devastated', 'devastation'],
            'angry': ['angry', 'rage', 'fury', 'irritated', 'annoyance', 'outraged', 'frustration', 'resentful', 'vexed', 'agitated'],
            'fear': ['fear', 'scared', 'afraid', 'terrified', 'terror', 'anxious', 'worried', 'apprehensive', 'horrified', 'horror'],
            'surprise': ['surprise', 'amazed', 'astonished', 'shocked', 'amazement', 'bewildered'],
            'disgust': ['disgust', 'repulsed', 'appalled', 'disgusted'],
            'neutral': ['neutral', 'calm', 'peaceful', 'serene', 'tranquil', 'indifferent', 'apathetic', 'boredom']
        }
    
    def map_to_basic_emotions(self, emotion):
        """Map complex emotions to basic categories"""
        emotion_lower = emotion.lower()
        
        for basic_emotion, variants in self.basic_emotions.items():
            if any(variant in emotion_lower for variant in variants):
                return basic_emotion
        
        return 'neutral'  # Default to neutral for unknown emotions
    
    def load_and_filter_data(self):
        """Load data and filter to basic emotions"""
        print("🔄 Loading and filtering for basic emotions...")
        
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        print(f"✅ Original dataset: {len(df)} samples")
        
        # Map to basic emotions
        df['basic_emotion'] = df['emotion'].apply(self.map_to_basic_emotions)
        
        # Count basic emotions
        emotion_counts = df['basic_emotion'].value_counts()
        print(f"📊 Basic emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"   {emotion}: {count} samples")
        
        print(f"✅ Filtered to {len(df)} samples with {len(self.basic_emotions)} basic emotions")
        return df
    
    def create_powerful_vectorizer(self):
        """Create high-performance text vectorizer"""
        print("🔧 Creating powerful text processor...")
        
        vectorizer = TfidfVectorizer(
            max_features=30000,          # High feature count for better representation
            ngram_range=(1, 4),         # Up to 4-grams for context
            stop_words='english',
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,            # Better for frequency scaling
            norm='l2',                   # L2 normalization for better performance
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )
        
        return vectorizer
    
    def create_powerful_ensemble(self):
        """Create powerful ensemble model"""
        print("🚀 Creating powerful ensemble model...")
        
        # High-performance individual models
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            bootstrap=False
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        lr = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=10.0,
            n_jobs=-1,
            solver='lbfgs'
        )
        
        svm = SVC(
            random_state=42,
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True
        )
        
        # Create weighted ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('random_forest', rf),
                ('gradient_boosting', gb),
                ('logistic_regression', lr),
                ('svm', svm)
            ],
            voting='soft',
            weights=[2, 2, 1, 1]  # Give more weight to tree-based models
        )
        
        return ensemble
    
    def train_optimized_model(self, test_size=0.2):
        """Train optimized model with maximum processing power"""
        print("🚀 Training optimized basic emotion model...")
        
        # Load and filter data
        df = self.load_and_filter_data()
        
        # Preprocess text
        print("🔄 Preprocessing text...")
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        # Remove problematic samples
        df = df[df['cleaned_text'].notna()]
        df = df[df['cleaned_text'].str.len() > 0]
        df = df.reset_index(drop=True)
        
        print(f"✅ Preprocessed: {len(df)} valid samples")
        
        # Prepare features and labels
        X = df['cleaned_text']
        y = df['basic_emotion']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"📊 Training: {len(X_train)} samples")
        print(f"📊 Testing: {len(X_test)} samples")
        
        # Create powerful vectorizer
        self.vectorizer = self.create_powerful_vectorizer()
        
        # Transform data
        print("🔄 Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"✅ Feature matrix: {X_train_tfidf.shape}")
        
        # Create and train powerful ensemble
        self.model = self.create_powerful_ensemble()
        
        print("🚀 Training ensemble model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        print("📊 Evaluating model...")
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # Detailed report
        print("\n📈 Classification Report:")
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        print(classification_report(y_test_labels, y_pred_labels))
        
        # Show confusion matrix summary
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix Summary:")
        emotions = self.label_encoder.classes_
        for i, emotion in enumerate(emotions):
            correct = cm[i, i]
            total = cm[i, :].sum()
            print(f"   {emotion}: {correct}/{total} correct ({correct/total*100:.1f}%)")
        
        return accuracy
    
    def save_optimized_model(self, model_path="optimized_basic_model.pkl", 
                           vectorizer_path="optimized_basic_vectorizer.pkl",
                           encoder_path="optimized_basic_encoder.pkl",
                           mappings_path="optimized_basic_mappings.pkl"):
        """Save optimized model"""
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save vectorizer
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save encoder
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save mappings
            mappings = {
                'basic_emotions': list(self.basic_emotions.keys()),
                'emotion_mappings': self.basic_emotions,
                'model_type': 'Optimized Ensemble',
                'num_classes': len(self.basic_emotions)
            }
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
            
            print(f"\n✅ Optimized model saved successfully!")
            print(f"   📁 Model: {model_path}")
            print(f"   📁 Vectorizer: {vectorizer_path}")
            print(f"   📁 Encoder: {encoder_path}")
            print(f"   📁 Mappings: {mappings_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

def main():
    """Main function for optimized basic emotion training"""
    print("🧠 OPTIMIZED BASIC EMOTION DETECTION")
    print("=" * 50)
    print("🎯 Focus: 7 Basic Emotions")
    print("⚡ Power: Maximum Processing")
    print("📈 Goal: Better Output Quality")
    print("=" * 50)
    
    # Initialize trainer
    trainer = OptimizedBasicEmotionTrainer()
    
    # Train optimized model
    accuracy = trainer.train_optimized_model(test_size=0.15)
    
    # Save model
    trainer.save_optimized_model()
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"🚀 Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("✅ Model ready for production!")

if __name__ == "__main__":
    main()
