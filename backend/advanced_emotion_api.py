"""
Advanced Emotion Detection with HuggingFace Transformers
Proper NLP-based emotion classification
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from typing import Dict, List, Optional

class AdvancedEmotionDetector:
    """Advanced emotion detector using HuggingFace transformers"""
    
    def __init__(self):
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.emotion_mapping = {
            'sadness': 'sadness',
            'joy': 'happiness', 
            'love': 'happiness',
            'anger': 'anger',
            'fear': 'fear',
            'surprise': 'surprise'
        }
        
    def load_model(self):
        """Load the transformer model"""
        try:
            print(f"🤖 Loading model: {self.model_name}")
            
            # Create emotion classification pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_emotion(self, text: str, include_debug: bool = False) -> Dict:
        """Detect emotion in text with proper NLP"""
        if not text or not text.strip():
            return {
                "error": "Please enter text",
                "emotion": None,
                "confidence": 0.0,
                "probabilities": {},
                "debug": {}
            }
        
        try:
            # Get model predictions
            results = self.pipeline(text.strip())
            
            # Process results
            probabilities = {}
            for result in results[0]:
                original_label = result['label']
                mapped_label = self.emotion_mapping.get(original_label, original_label)
                probability = result['score']
                
                # Accumulate probabilities for mapped emotions
                if mapped_label in probabilities:
                    probabilities[mapped_label] += probability
                else:
                    probabilities[mapped_label] = probability
            
            # Ensure all required emotions are present
            required_emotions = ['sadness', 'happiness', 'anger', 'fear', 'surprise']
            for emotion in required_emotions:
                if emotion not in probabilities:
                    probabilities[emotion] = 0.0
            
            # Add neutral (inverse of max emotion)
            max_prob = max(probabilities.values())
            probabilities['neutral'] = max(0.0, 1.0 - max_prob)
            
            # Sort by probability
            sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            # Get primary emotion
            primary_emotion, primary_confidence = sorted_emotions[0]
            
            # Adjust confidence for very short inputs
            if len(text.strip()) < 5:
                primary_confidence *= 0.7
            
            # Prepare debug info
            debug_info = {}
            if include_debug:
                debug_info = {
                    "raw_scores": results[0],
                    "mapped_scores": probabilities,
                    "input_length": len(text.strip()),
                    "model_used": self.model_name,
                    "threshold_applied": len(text.strip()) < 5
                }
            
            return {
                "emotion": primary_emotion,
                "confidence": min(primary_confidence, 1.0),
                "probabilities": probabilities,
                "all_emotions_sorted": sorted_emotions,
                "debug": debug_info,
                "error": None
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "emotion": None,
                "confidence": 0.0,
                "probabilities": {},
                "debug": {}
            }

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize detector
detector = AdvancedEmotionDetector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": detector.pipeline is not None})

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        include_debug = data.get('debug', False)
        
        result = detector.detect_emotion(text, include_debug)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}",
            "emotion": None,
            "confidence": 0.0,
            "probabilities": {}
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_name": detector.model_name,
        "emotions_supported": ['sadness', 'happiness', 'anger', 'fear', 'surprise', 'neutral'],
        "emotion_mapping": detector.emotion_mapping,
        "model_loaded": detector.pipeline is not None
    })

if __name__ == '__main__':
    # Load model before starting server
    if detector.load_model():
        print("🚀 Starting Advanced Emotion Detection API...")
        app.run(host='0.0.0.0', port=8000, debug=False)
    else:
        print("❌ Failed to load model. Exiting...")
