"""
Training script for emotion detection model
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import TextPreprocessor

class EmotionTrainer:
    """Class for training emotion detection models"""
    
    def __init__(self, dataset_path="../../dataset/emotions_expanded.csv"):
        self.dataset_path = dataset_path
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']
        
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Dataset loaded: {len(df)} samples")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the text data"""
        print("🔄 Preprocessing text data...")
        
        # Clean the text
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"✅ Preprocessing complete: {len(df)} valid samples")
        return df
    
    def train_model(self, model_type='naive_bayes', test_size=0.2, max_features=5000, ngram_range=(1, 2)):
        """
        Train the emotion detection model
        
        Args:
            model_type (str): Type of model ('naive_bayes' or 'logistic_regression')
            test_size (float): Proportion of data for testing
            max_features (int): Maximum number of features for TF-IDF
            ngram_range (tuple): N-gram range for TF-IDF
        """
        print(f"🚀 Training {model_type} model...")
        
        # Load and preprocess data
        df = self.load_data()
        if df is None:
            return False
        
        df = self.preprocess_data(df)
        
        # Prepare features and labels
        X = df['cleaned_text']
        y = df['emotion']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        # Fit and transform the data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            print(f"❌ Unknown model type: {model_type}")
            return False
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model training complete!")
        print(f"📊 Accuracy: {accuracy:.3f}")
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return True
    
    def save_model(self, model_path="model.pkl", vectorizer_path="vectorizer.pkl", mappings_path="model_mappings.pkl"):
        """Save the trained model and components"""
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save vectorizer
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save mappings
            mappings = {
                'emotions': self.emotions,
                'model_type': type(self.model).__name__
            }
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
            
            print(f"✅ Model saved successfully!")
            print(f"   📁 Model: {model_path}")
            print(f"   📁 Vectorizer: {vectorizer_path}")
            print(f"   📁 Mappings: {mappings_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

def main():
    """Main training function"""
    print("🧠 Emotion Detection Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = EmotionTrainer()
    
    # Train model
    success = trainer.train_model(
        model_type='logistic_regression',
        test_size=0.2,
        max_features=8000,
        ngram_range=(1, 3)
    )
    
    if success:
        # Save model
        trainer.save_model()
        print("\n🎉 Training completed successfully!")
        print("🚀 Model is ready for predictions!")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()
