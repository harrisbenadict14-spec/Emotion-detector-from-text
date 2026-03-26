"""
Working Emotion Detection API
Fixed logic with proper emotion classification
"""

import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List

class WorkingEmotionDetector:
    """Working emotion detector with proper logic"""
    
    def __init__(self):
        # Enhanced emotion keywords with proper weights
        self.emotion_keywords = {
            'sadness': {
                'words': ['sad', 'crying', 'cry', 'tears', 'lonely', 'depressed', 'unhappy', 'miserable', 'grief', 'sorrow', 'hurt', 'pain', 'broken', 'lost', 'missing', 'alone'],
                'weight': 2.0
            },
            'happiness': {
                'words': ['happy', 'joy', 'excited', 'love', 'wonderful', 'amazing', 'great', 'fantastic', 'excellent', 'smile', 'laugh', 'celebrate', 'won', 'success', 'proud', 'thrilled'],
                'weight': 2.0
            },
            'anger': {
                'words': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'frustrated', 'irritated', 'outraged', 'stupid', 'ridiculous', 'unfair', 'wrong'],
                'weight': 2.0
            },
            'fear': {
                'words': ['scared', 'afraid', 'fear', 'terrified', 'anxious', 'worried', 'nervous', 'panic', 'danger', 'threat', 'scary', 'frightened'],
                'weight': 2.0
            },
            'surprise': {
                'words': ['surprised', 'amazed', 'shocked', 'astonished', 'wow', 'sudden', 'unexpected', 'believe', 'incredible', 'unbelievable'],
                'weight': 2.0
            }
        }
        
        # Context patterns
        self.context_patterns = {
            'sadness': [
                r'i am\s+(?:very\s+)?(?:crying|sad|depressed)',
                r'feel(?:ing)?\s+(?:sad|lonely|down)',
                r'making\s+me\s+cry',
                r'can\'t\s+stop\s+crying'
            ],
            'happiness': [
                r'i\s+(?:just\s+)?(?:won|got)',
                r'so\s+(?:happy|excited|thrilled)',
                r'love\s+(?:this|it|you)',
                r'(?:great|wonderful|amazing)\s+(?:day|news|time)'
            ],
            'anger': [
                r'i\s+am\s+(?:so\s+)?(?:angry|mad)',
                r'(?:stupid|ridiculous|unfair)',
                r'makes\s+me\s+(?:angry|mad)',
                r'can\'t\s+believe\s+(?:this|that)'
            ],
            'fear': [
                r'i\'?m\s+(?:scared|afraid|frightened)',
                r'fear\s+(?:of|for)',
                r'worried\s+about',
                r'(?:scary|terrifying|frightening)'
            ]
        }
    
    def detect_emotion(self, text: str, include_debug: bool = False) -> Dict:
        """Detect emotion with proper logic"""
        if not text or not text.strip():
            return {
                "error": "Please enter text",
                "emotion": None,
                "confidence": 0.0,
                "probabilities": {},
                "debug": {}
            }
        
        try:
            text_clean = text.lower().strip()
            
            # Initialize probabilities
            probabilities = {
                'sadness': 0.0,
                'happiness': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'surprise': 0.0,
                'neutral': 0.1  # Base neutral probability
            }
            
            # Context pattern matching (highest priority)
            for emotion, patterns in self.context_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_clean):
                        probabilities[emotion] += 0.8
                        break
            
            # Keyword matching
            word_count = len(text_clean.split())
            for emotion, config in self.emotion_keywords.items():
                matches = 0
                for word in config['words']:
                    if word in text_clean:
                        matches += 1
                        # Weight based on position and emphasis
                        if word in text_clean.split()[:3]:  # In first 3 words
                            matches += 0.5
                        if text_clean.count(word) > 1:  # Repeated
                            matches += 0.3
                
                # Calculate probability
                if matches > 0:
                    base_prob = (matches * config['weight']) / word_count
                    probabilities[emotion] += min(base_prob, 0.9)
            
            # Normalize probabilities
            total_prob = sum(probabilities.values())
            if total_prob > 0:
                for emotion in probabilities:
                    probabilities[emotion] = probabilities[emotion] / total_prob
            
            # Sort emotions by probability
            sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            # Get primary emotion
            primary_emotion, primary_confidence = sorted_emotions[0]
            
            # Adjust confidence based on text length
            if word_count < 3:
                primary_confidence *= 0.7
            elif word_count > 10:
                primary_confidence *= 1.1
            
            # Cap confidence at 1.0
            primary_confidence = min(primary_confidence, 1.0)
            
            # Prepare debug info
            debug_info = {}
            if include_debug:
                debug_info = {
                    "cleaned_text": text_clean,
                    "word_count": word_count,
                    "raw_probabilities": probabilities,
                    "sorted_emotions": sorted_emotions,
                    "context_matches": {emotion: any(re.search(pattern, text_clean) for pattern in patterns) 
                                     for emotion, patterns in self.context_patterns.items()},
                    "keyword_matches": {emotion: [word for word in config['words'] if word in text_clean] 
                                      for emotion, config in self.emotion_keywords.items()}
                }
            
            return {
                "emotion": primary_emotion,
                "confidence": primary_confidence,
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
detector = WorkingEmotionDetector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True})

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
        "model_name": "Working Emotion Detector v2.0",
        "emotions_supported": ['sadness', 'happiness', 'anger', 'fear', 'surprise', 'neutral'],
        "model_type": "Rule-based with context patterns",
        "model_loaded": True
    })

if __name__ == '__main__':
    print("🚀 Starting Working Emotion Detection API...")
    app.run(host='0.0.0.0', port=8000, debug=False)
