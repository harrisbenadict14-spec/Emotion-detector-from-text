"""
Core 6 Emotions Detection - Standard Ekman Emotions with Maximum Accuracy
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

class CoreEmotionTrainer:
    """Trainer for the 6 core Ekman emotions"""
    
    def __init__(self, dataset_path="../../dataset/comprehensive_emotions.csv"):
        self.dataset_path = dataset_path
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Map to the 6 core Ekman emotions
        self.core_emotions_mapping = {
            'happiness': ['happy', 'joy', 'excitement', 'delight', 'elated', 'jubilant', 'ecstatic', 'euphoric', 'euphoria', 'blissful', 'bliss', 'overjoyed', 'thrilled', 'contentment', 'gratitude', 'pride'],
            'sadness': ['sad', 'sorrow', 'melancholy', 'disappointment', 'grief', 'depressed', 'melancholic', 'devastated', 'devastation', 'shame', 'guilt', 'loneliness', 'hurt'],
            'anger': ['angry', 'rage', 'fury', 'irritated', 'annoyance', 'outraged', 'frustration', 'resentful', 'vexed', 'agitated', 'irritation'],
            'fear': ['fear', 'scared', 'afraid', 'terrified', 'terror', 'anxious', 'worried', 'apprehensive', 'horrified', 'horror', 'anxiety'],
            'surprise': ['surprise', 'amazed', 'astonished', 'shocked', 'amazement', 'bewildered', 'curiosity'],
            'disgust': ['disgust', 'repulsed', 'appalled', 'disgusted']
        }
    
    def map_to_core_emotion(self, emotion):
        """Map complex emotions to 6 core emotions"""
        emotion_lower = emotion.lower()
        
        for core_emotion, variants in self.core_emotions_mapping.items():
            if any(variant in emotion_lower for variant in variants):
                return core_emotion
        
        # Default mapping for unmapped emotions
        emotion_lower = emotion.lower()
        if any(word in emotion_lower for word in ['neutral', 'calm', 'peaceful', 'boredom']):
            return 'sadness'  # Map neutral-ish to sadness
        elif any(word in emotion_lower for word in ['confusion', 'baffled']):
            return 'surprise'
        else:
            return 'sadness'  # Default fallback
    
    def load_and_map_data(self):
        """Load data and map to 6 core emotions"""
        print("🔄 Loading and mapping to 6 core emotions...")
        
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        print(f"✅ Original dataset: {len(df)} samples")
        
        # Map to core emotions
        df['core_emotion'] = df['emotion'].apply(self.map_to_core_emotion)
        
        # Count core emotions
        emotion_counts = df['core_emotion'].value_counts()
        print(f"\n📊 Core 6 Emotion Distribution:")
        for emotion, count in emotion_counts.items():
            print(f"   {emotion.capitalize()}: {count} samples")
        
        print(f"\n✅ Mapped to {len(df)} samples with 6 core emotions")
        return df
    
    def create_optimal_vectorizer(self):
        """Create optimal vectorizer for 6 emotions"""
        print("🔧 Creating optimal text processor for 6 emotions...")
        
        vectorizer = TfidfVectorizer(
            max_features=25000,          # Optimal for 6 classes
            ngram_range=(1, 3),         # 1-3 grams for balance
            stop_words='english',
            min_df=1,
            max_df=0.85,
            sublinear_tf=True,
            norm='l2',
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )
        
        return vectorizer
    
    def create_optimal_ensemble(self):
        """Create optimal ensemble for 6 emotions"""
        print("🚀 Creating optimal ensemble for 6 core emotions...")
        
        # Optimized models for 6-class classification
        rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            bootstrap=False
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.15,
            max_depth=5,
            random_state=42
        )
        
        lr = LogisticRegression(
            random_state=42,
            max_iter=1500,
            C=15.0,
            solver='lbfgs'
        )
        
        svm = SVC(
            random_state=42,
            kernel='rbf',
            C=8.0,
            gamma='scale',
            probability=True
        )
        
        # Ensemble with equal weights for 6 classes
        ensemble = VotingClassifier(
            estimators=[
                ('random_forest', rf),
                ('gradient_boosting', gb),
                ('logistic_regression', lr),
                ('svm', svm)
            ],
            voting='soft',
            weights=[1.5, 1.5, 1, 1]  # Slightly favor tree models
        )
        
        return ensemble
    
    def train_core_model(self, test_size=0.15):
        """Train model on 6 core emotions"""
        print("🚀 Training 6 Core Emotions Model...")
        
        # Load and map data
        df = self.load_and_map_data()
        
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
        y = df['core_emotion']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"📊 Training: {len(X_train)} samples")
        print(f"📊 Testing: {len(X_test)} samples")
        
        # Create optimal vectorizer
        self.vectorizer = self.create_optimal_vectorizer()
        
        # Transform data
        print("🔄 Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"✅ Feature matrix: {X_train_tfidf.shape}")
        
        # Create and train ensemble
        self.model = self.create_optimal_ensemble()
        
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
        print(classification_report(y_test_labels, y_pred_labels, digits=4))
        
        # Show confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        emotions = self.label_encoder.classes_
        print("Predicted →")
        print("Actual ↓", end="")
        for emotion in emotions:
            print(f" {emotion[:8]}", end="")
        print()
        for i, true_emotion in enumerate(emotions):
            print(f"{true_emotion[:8]}", end="")
            for j in range(len(emotions)):
                print(f" {cm[i][j]:6}", end="")
            print()
        
        return accuracy
    
    def save_core_model(self, model_path="core_6_emotions_model.pkl", 
                      vectorizer_path="core_6_emotions_vectorizer.pkl",
                      encoder_path="core_6_emotions_encoder.pkl",
                      mappings_path="core_6_emotions_mappings.pkl"):
        """Save core emotions model"""
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
                'core_emotions': ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                'emotion_mappings': self.core_emotions_mapping,
                'model_type': 'Core 6 Emotions Ensemble',
                'num_classes': 6,
                'description': 'Ekman\'s 6 basic emotions'
            }
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
            
            print(f"\n✅ Core 6 Emotions model saved successfully!")
            print(f"   📁 Model: {model_path}")
            print(f"   📁 Vectorizer: {vectorizer_path}")
            print(f"   📁 Encoder: {encoder_path}")
            print(f"   📁 Mappings: {mappings_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

def main():
    """Main function for core 6 emotions training"""
    print("🧠 CORE 6 EMOTIONS DETECTION")
    print("=" * 40)
    print("😊 Happiness (Joy)")
    print("😢 Sadness") 
    print("😠 Anger")
    print("😨 Fear")
    print("😲 Surprise")
    print("🤢 Disgust")
    print("=" * 40)
    print("📚 Based on Ekman's 6 Basic Emotions")
    print("⚡ Maximum Processing Power")
    print("🎯 Optimal for 6-Class Classification")
    print("=" * 40)
    
    # Initialize trainer
    trainer = CoreEmotionTrainer()
    
    # Train core model
    accuracy = trainer.train_core_model(test_size=0.12)
    
    # Save model
    trainer.save_core_model()
    
    print(f"\n🎉 CORE 6 EMOTIONS TRAINING COMPLETED!")
    print(f"🚀 Final Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("✅ Model ready for production!")

if __name__ == "__main__":
    main()
