"""
Prediction module for emotion detection
"""

import pickle
import numpy as np
import os
from preprocess import TextPreprocessor

class EmotionPredictor:
    """Class for making emotion predictions"""
    
    def __init__(self, model_path="model/model.pkl", vectorizer_path="model/vectorizer.pkl", mappings_path="model/model_mappings.pkl"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.mappings_path = mappings_path
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.mappings = None
        self.emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']
        
        # Load model components
        self.load_model()
    
    def load_model(self):
        """Load the trained model and components"""
        try:
            # Load model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Model loaded: {self.model_path}")
            else:
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            # Load vectorizer
            if os.path.exists(self.vectorizer_path):
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print(f"✅ Vectorizer loaded: {self.vectorizer_path}")
            else:
                print(f"❌ Vectorizer file not found: {self.vectorizer_path}")
                return False
            
            # Load mappings
            if os.path.exists(self.mappings_path):
                with open(self.mappings_path, 'rb') as f:
                    self.mappings = pickle.load(f)
                print(f"✅ Mappings loaded: {self.mappings_path}")
            else:
                print(f"❌ Mappings file not found: {self.mappings_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, text, include_probabilities=True):
        """
        Predict emotion for a single text
        
        Args:
            text (str): Input text
            include_probabilities (bool): Whether to include probability distribution
            
        Returns:
            dict: Prediction results
        """
        if not self.model or not self.vectorizer:
            return {
                'error': 'Model not loaded',
                'emotion': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            
            if not cleaned_text:
                return {
                    'error': 'Empty or invalid text after preprocessing',
                    'emotion': 'unknown',
                    'confidence': 0.0
                }
            
            # Vectorize text
            text_vector = self.vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.model.predict(text_vector)[0]
            
            # Get confidence score
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_vector)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 1.0
                probabilities = None
            
            # Prepare result
            result = {
                'emotion': prediction,
                'confidence': float(confidence),
                'cleaned_text': cleaned_text
            }
            
            # Add probabilities if requested
            if include_probabilities and probabilities is not None:
                prob_dict = {}
                for i, emotion in enumerate(self.model.classes_):
                    prob_dict[emotion] = float(probabilities[i])
                
                # Ensure all emotions are included
                for emotion in self.emotions:
                    if emotion not in prob_dict:
                        prob_dict[emotion] = 0.0
                
                result['probabilities'] = prob_dict
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'emotion': 'unknown',
                'confidence': 0.0
            }
    
    def predict_batch(self, texts, include_probabilities=True):
        """
        Predict emotions for multiple texts
        
        Args:
            texts (list): List of input texts
            include_probabilities (bool): Whether to include probability distributions
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict(text, include_probabilities)
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.model:
            return {
                'model_loaded': False,
                'error': 'Model not loaded'
            }
        
        info = {
            'model_loaded': True,
            'model_type': type(self.model).__name__,
            'vectorizer': type(self.vectorizer).__name__,
            'supported_emotions': self.emotions,
            'features': self.vectorizer.get_feature_names_out().shape[0] if hasattr(self.vectorizer, 'get_feature_names_out') else 'Unknown'
        }
        
        if self.mappings:
            info.update(self.mappings)
        
        return info

def main():
    """Test the predictor"""
    print("🧠 Emotion Detection Predictor Test")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EmotionPredictor()
    
    # Test predictions
    test_texts = [
        "I am so happy today!",
        "I feel really sad and depressed",
        "This makes me so angry",
        "I'm scared and worried",
        "I'm completely surprised by this news",
        "I feel neutral about this situation"
    ]
    
    for text in test_texts:
        result = predictor.predict(text, include_probabilities=True)
        
        print(f"\n📝 Text: {text}")
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"😊 Emotion: {result['emotion'].upper()}")
            print(f"📈 Confidence: {result['confidence']:.1%}")
            if 'probabilities' in result:
                print("📊 Probabilities:")
                for emotion, prob in result['probabilities'].items():
                    print(f"   {emotion}: {prob:.1%}")

if __name__ == "__main__":
    main()
