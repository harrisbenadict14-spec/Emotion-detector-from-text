"""
Demo script to test the Emotion Detection API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the emotion detection API with sample texts"""
    
    # Sample texts for testing
    test_texts = [
        "I am so happy and excited today!",
        "I feel really sad and depressed about the news",
        "This makes me so angry and frustrated",
        "I'm scared and worried about what might happen",
        "I'm completely surprised by this unexpected news",
        "I feel neutral about this situation",
        "Today was absolutely amazing! I got the job I've been dreaming about for months.",
        "The funeral was so emotional, I can't stop crying.",
        "I'm furious that someone stole my wallet from my car.",
        "I have a terrible fear of heights and I'm scared to go on the roller coaster."
    ]
    
    print("🧠 AI Emotion Detector - API Demo")
    print("=" * 50)
    print()
    
    # Test health endpoint
    print("📡 Testing API Health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API Status: {health_data.get('status')}")
            print(f"✅ Model Loaded: {health_data.get('model_loaded')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the backend server is running on http://localhost:8000")
        return
    
    print()
    
    # Test model info
    print("🤖 Getting Model Information...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"📊 Model Type: {model_info.get('model_type')}")
            print(f"🔤 Vectorizer: {model_info.get('vectorizer')}")
            print(f"🎯 Features: {model_info.get('features')}")
            print(f"😊 Supported Emotions: {', '.join(model_info.get('supported_emotions', []))}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting model info: {e}")
    
    print()
    
    # Test predictions
    print("🔮 Testing Emotion Predictions...")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            # Make prediction request
            payload = {
                "text": text,
                "include_probabilities": True
            }
            
            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                emotion = result.get('emotion', 'unknown')
                confidence = result.get('confidence', 0)
                probabilities = result.get('probabilities', {})
                
                print(f"   😊 Emotion: {emotion.upper()}")
                print(f"   📈 Confidence: {confidence:.1%}")
                
                # Show top 3 probabilities
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
                prob_text = ", ".join([f"{emo}: {prob:.1%}" for emo, prob in sorted_probs])
                print(f"   📊 Top Probabilities: {prob_text}")
                
            else:
                print(f"   ❌ Prediction failed: {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    print()
    print("=" * 50)
    print("🎉 Demo completed!")
    print()
    print("💡 Tips:")
    print("   • Open frontend/index.html in your browser for the full UI")
    print("   • The API supports batch processing via /predict-batch")
    print("   • Check the API documentation at http://localhost:8000/docs")
    print("   • Results are saved to history in the frontend")

if __name__ == "__main__":
    test_api()
