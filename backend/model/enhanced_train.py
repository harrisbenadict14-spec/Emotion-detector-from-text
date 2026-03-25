"""
Enhanced training script for emotion detection model with 100% accuracy goal
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from preprocess import TextPreprocessor

class EnhancedEmotionTrainer:
    """Enhanced class for training emotion detection models with 100% accuracy goal"""
    
    def __init__(self, dataset_path="../../dataset/comprehensive_emotions.csv"):
        self.dataset_path = dataset_path
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.emotions = []
        
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Dataset loaded: {len(df)} samples")
            print(f"📊 Unique emotions: {df['emotion'].nunique()}")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def augment_text(self, text, num_augmentations=3):
        """Generate augmented versions of text"""
        augmented_texts = [text]
        
        # Synonym replacement (simplified version)
        synonyms = {
            'happy': ['joyful', 'glad', 'cheerful', 'delighted', 'pleased'],
            'sad': ['unhappy', 'sorrowful', 'depressed', 'melancholy', 'gloomy'],
            'angry': ['furious', 'mad', 'irritated', 'annoyed', 'outraged'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious', 'worried'],
            'surprise': ['amazed', 'astonished', 'shocked', 'stunned'],
            'excited': ['thrilled', 'enthusiastic', 'eager', 'exhilarated'],
            'calm': ['peaceful', 'serene', 'tranquil', 'relaxed'],
            'confused': ['bewildered', 'puzzled', 'perplexed', 'disoriented']
        }
        
        for _ in range(num_augmentations):
            aug_text = text
            
            # Random word variations
            for word, syns in synonyms.items():
                if word in aug_text.lower() and syns:
                    if random.random() < 0.3:  # 30% chance to replace
                        replacement = random.choice(syns)
                        aug_text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, aug_text, flags=re.IGNORECASE)
            
            # Add intensity modifiers
            if random.random() < 0.2:
                intensifiers = ['very', 'extremely', 'really', 'quite', 'so', 'incredibly']
                aug_text = random.choice(intensifiers) + ' ' + aug_text
            
            # Change sentence structure slightly
            if random.random() < 0.2:
                if aug_text.startswith('I feel'):
                    aug_text = aug_text.replace('I feel', 'Feeling')
                elif aug_text.startswith('I\'m'):
                    aug_text = aug_text.replace('I\'m', 'I am')
            
            # Only add if it's different from original
            if aug_text != text and aug_text not in augmented_texts:
                augmented_texts.append(aug_text)
        
        # Ensure we have at least one augmentation (even if it's just the original)
        if len(augmented_texts) == 1:
            # Create a simple variation
            simple_aug = text + "!"
            if simple_aug != text:
                augmented_texts.append(simple_aug)
            else:
                augmented_texts.append("I " + text)
        
        return augmented_texts
    
    def balance_dataset(self, df, target_samples_per_class=10):
        """Balance dataset by augmenting underrepresented classes"""
        print("🔄 Balancing dataset with data augmentation...")
        
        emotion_counts = df['emotion'].value_counts()
        balanced_df = df.copy()
        
        for emotion, count in emotion_counts.items():
            if count < target_samples_per_class:
                emotion_texts = df[df['emotion'] == emotion]['text'].tolist()
                needed = target_samples_per_class - count
                
                print(f"   📈 Augmenting '{emotion}': {count} -> {target_samples_per_class}")
                
                augmented_samples = []
                for i in range(needed):
                    original_text = emotion_texts[i % len(emotion_texts)]
                    augmented_texts = self.augment_text(original_text, 1)
                    if len(augmented_texts) > 1:
                        augmented = augmented_texts[1]  # Get first augmentation
                    else:
                        augmented = augmented_texts[0]  # Fallback to original
                    
                    # Ensure augmented text is not empty
                    if augmented.strip():
                        augmented_samples.append({'text': augmented, 'emotion': emotion})
                
                if augmented_samples:
                    aug_df = pd.DataFrame(augmented_samples)
                    balanced_df = pd.concat([balanced_df, aug_df], ignore_index=True)
        
        print(f"✅ Dataset balanced: {len(balanced_df)} total samples")
        return balanced_df
    
    def preprocess_data(self, df):
        """Preprocess the text data"""
        print("🔄 Preprocessing text data...")
        
        # Clean the text
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        # Remove empty texts and NaN values
        df = df[df['cleaned_text'].notna()]  # Remove NaN
        df = df[df['cleaned_text'].str.len() > 0]  # Remove empty strings
        df = df[df['cleaned_text'].str.strip() != '']  # Remove whitespace-only strings
        
        # Reset index to avoid issues
        df = df.reset_index(drop=True)
        
        print(f"✅ Preprocessing complete: {len(df)} valid samples")
        return df
    
    def create_ensemble_model(self):
        """Create an ensemble of multiple models"""
        print("🔧 Creating ensemble model...")
        
        # Individual models
        nb_model = MultinomialNB(alpha=0.1)
        lr_model = LogisticRegression(random_state=42, max_iter=2000, C=10)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
        svm_model = SVC(random_state=42, kernel='linear', probability=True, C=5)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('naive_bayes', nb_model),
                ('logistic_regression', lr_model),
                ('random_forest', rf_model),
                ('svm', svm_model)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def train_model(self, use_ensemble=True, balance_data=True, test_size=0.1):
        """
        Train the emotion detection model with enhanced techniques
        
        Args:
            use_ensemble (bool): Whether to use ensemble model
            balance_data (bool): Whether to balance the dataset
            test_size (float): Proportion of data for testing
        """
        print(f"🚀 Enhanced Training - Ensemble: {use_ensemble}, Balanced: {balance_data}")
        
        # Load and preprocess data
        df = self.load_data()
        if df is None:
            return False
        
        df = self.preprocess_data(df)
        
        # Balance dataset if requested
        if balance_data:
            df = self.balance_dataset(df, target_samples_per_class=15)
        
        # Prepare features and labels
        X = df['cleaned_text']
        y = df['emotion']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.emotions = self.label_encoder.classes_.tolist()
        
        # Split data with stratification (only if test size allows)
        if test_size * len(df) >= len(self.emotions):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        # Create enhanced TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Fit and transform the data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Apply SMOTE for additional balancing
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
        
        print(f"📊 After SMOTE - Training samples: {X_train_balanced.shape[0]}")
        
        # Create and train model
        if use_ensemble:
            self.model = self.create_ensemble_model()
        else:
            self.model = LogisticRegression(random_state=42, max_iter=2000, C=10)
        
        # Train with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_balanced, y_train_balanced, cv=cv, scoring='accuracy')
        
        print(f"📈 Cross-validation scores: {cv_scores}")
        print(f"📊 Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train on full dataset
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model training complete!")
        print(f"📊 Test Accuracy: {accuracy:.3f}")
        
        # Detailed classification report
        print("\n📈 Detailed Classification Report:")
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
        
        return accuracy >= 0.95  # Return True if accuracy is 95% or higher
    
    def save_model(self, model_path="enhanced_model.pkl", vectorizer_path="enhanced_vectorizer.pkl", 
                   encoder_path="label_encoder.pkl", mappings_path="enhanced_mappings.pkl"):
        """Save the trained model and components"""
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save vectorizer
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save label encoder
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save mappings
            mappings = {
                'emotions': self.emotions,
                'model_type': type(self.model).__name__,
                'num_classes': len(self.emotions)
            }
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
            
            print(f"✅ Enhanced model saved successfully!")
            print(f"   📁 Model: {model_path}")
            print(f"   📁 Vectorizer: {vectorizer_path}")
            print(f"   📁 Label Encoder: {encoder_path}")
            print(f"   📁 Mappings: {mappings_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

def main():
    """Main enhanced training function"""
    print("🧠 Enhanced Emotion Detection Model Training - Target: 100% Accuracy")
    print("=" * 70)
    
    # Initialize trainer
    trainer = EnhancedEmotionTrainer()
    
    # Try multiple configurations for best accuracy
    configurations = [
        {'use_ensemble': True, 'balance_data': True, 'test_size': 0.2},
        {'use_ensemble': True, 'balance_data': True, 'test_size': 0.15},
        {'use_ensemble': False, 'balance_data': True, 'test_size': 0.2},
    ]
    
    best_accuracy = 0
    best_config = None
    
    for i, config in enumerate(configurations, 1):
        print(f"\n🔄 Configuration {i}: {config}")
        print("-" * 50)
        
        success = trainer.train_model(**config)
        
        if success:
            # Save this model
            model_path = f"enhanced_model_v{i}.pkl"
            vectorizer_path = f"enhanced_vectorizer_v{i}.pkl"
            encoder_path = f"label_encoder_v{i}.pkl"
            mappings_path = f"enhanced_mappings_v{i}.pkl"
            
            trainer.save_model(model_path, vectorizer_path, encoder_path, mappings_path)
    
    print("\n🎉 Enhanced training completed!")
    print("🚀 Check the saved models for the best performing configuration!")

if __name__ == "__main__":
    main()
