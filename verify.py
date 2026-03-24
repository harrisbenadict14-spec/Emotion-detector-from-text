"""
Final verification script for AI Emotion Detector
Tests all components and provides status report
"""

import os
import sys
import subprocess
import requests
import time
import json

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_status(status, message):
    """Print formatted status"""
    icon = "✅" if status else "❌"
    print(f"{icon} {message}")

def check_file_exists(file_path, description):
    """Check if file exists"""
    exists = os.path.exists(file_path)
    print_status(exists, f"{description}: {file_path}")
    return exists

def check_directory_structure():
    """Verify project structure"""
    print_header("📁 Checking Project Structure")
    
    required_files = [
        ("backend/main.py", "Backend API server"),
        ("backend/requirements.txt", "Backend dependencies"),
        ("backend/model/train.py", "Model training script"),
        ("backend/model/predict.py", "Prediction module"),
        ("backend/model/preprocess.py", "Text preprocessing"),
        ("backend/model/model.pkl", "Trained model"),
        ("backend/model/vectorizer.pkl", "TF-IDF vectorizer"),
        ("backend/model/model_mappings.pkl", "Label mappings"),
        ("frontend/index.html", "Frontend HTML"),
        ("frontend/style.css", "Frontend styling"),
        ("frontend/script.js", "Frontend JavaScript"),
        ("dataset/emotions.csv", "Basic dataset"),
        ("dataset/emotions_expanded.csv", "Expanded dataset"),
        ("README.md", "Documentation"),
        ("MANUAL.md", "User manual"),
        ("start.bat", "Windows startup script"),
        ("start.sh", "Linux/Mac startup script"),
        ("demo.py", "Demo script")
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def check_python_packages():
    """Check if required Python packages are installed"""
    print_header("🐍 Checking Python Packages")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "scikit-learn",
        "nltk",
        "pandas",
        "numpy",
        "pydantic"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print_status(True, f"Package '{package}' is installed")
        except ImportError:
            print_status(False, f"Package '{package}' is NOT installed")
            all_installed = False
    
    return all_installed

def test_model_training():
    """Test if model can be loaded"""
    print_header("🤖 Testing ML Model")
    
    try:
        # Change to model directory
        os.chdir("backend/model")
        
        # Try to import and test the predictor
        sys.path.append('.')
        from predict import EmotionPredictor
        
        predictor = EmotionPredictor()
        
        # Test prediction
        test_text = "I am so happy today!"
        result = predictor.predict(test_text)
        
        print_status(True, "Model loaded successfully")
        print_status(True, f"Test prediction: '{test_text}' -> {result['emotion']} ({result['confidence']:.1%})")
        
        # Return to original directory
        os.chdir("../..")
        return True
        
    except Exception as e:
        print_status(False, f"Model error: {str(e)}")
        try:
            os.chdir("../..")
        except:
            pass
        return False

def test_api_server():
    """Test if API server is running"""
    print_header("🌐 Testing API Server")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print_status(True, f"API Status: {health_data.get('status', 'unknown')}")
            print_status(True, f"Model Loaded: {health_data.get('model_loaded', False)}")
            
            # Test prediction endpoint
            payload = {
                "text": "I am so happy today!",
                "include_probabilities": True
            }
            
            pred_response = requests.post(
                "http://localhost:8000/predict",
                json=payload,
                timeout=5
            )
            
            if pred_response.status_code == 200:
                result = pred_response.json()
                print_status(True, f"Prediction API working: {result.get('emotion')} ({result.get('confidence', 0):.1%})")
                return True
            else:
                print_status(False, f"Prediction API failed: {pred_response.status_code}")
                return False
        else:
            print_status(False, f"Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_status(False, "API server not running on http://localhost:8000")
        print_status(False, "Start the server with: cd backend && python main.py")
        return False
    except Exception as e:
        print_status(False, f"API test error: {str(e)}")
        return False

def test_frontend():
    """Test frontend files"""
    print_header("🎨 Testing Frontend")
    
    # Check if frontend files exist and have content
    frontend_files = [
        ("frontend/index.html", "HTML structure"),
        ("frontend/style.css", "CSS styling"),
        ("frontend/script.js", "JavaScript functionality")
    ]
    
    all_good = True
    for file_path, description in frontend_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 100:  # Basic content check
                    print_status(True, f"{description} has content ({len(content)} bytes)")
                else:
                    print_status(False, f"{description} seems empty or too small")
                    all_good = False
        else:
            print_status(False, f"{description} file missing")
            all_good = False
    
    return all_good

def provide_instructions():
    """Provide next steps"""
    print_header("🚀 Next Steps")
    
    print("\n📋 TO RUN THE APPLICATION:")
    print("\n1. EASY WAY (Recommended):")
    print("   • Double-click: start.bat (Windows)")
    print("   • Or run: ./start.sh (Mac/Linux)")
    
    print("\n2. MANUAL WAY:")
    print("   • cd backend")
    print("   • pip install -r requirements.txt")
    print("   • cd model && python train.py && cd ..")
    print("   • python main.py")
    print("   • Open frontend/index.html in browser")
    
    print("\n3. TEST THE API:")
    print("   • Run: python demo.py")
    print("   • Or visit: http://localhost:8000")
    
    print("\n4. USE THE WEB APP:")
    print("   • Open frontend/index.html")
    print("   • Type text and click 'Analyze Emotion'")
    print("   • View results and save to history")
    
    print("\n📖 FOR HELP:")
    print("   • Read MANUAL.md for detailed instructions")
    print("   • Check README.md for technical details")
    print("   • Visit http://localhost:8000/docs for API documentation")

def main():
    """Main verification function"""
    print_header("🧠 AI Emotion Detector - Final Verification")
    print("Checking all components and providing setup instructions...")
    
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run all checks
    checks = [
        ("Project Structure", check_directory_structure),
        ("Python Packages", check_python_packages),
        ("ML Model", test_model_training),
        ("Frontend Files", test_frontend),
        ("API Server", test_api_server)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {str(e)}")
            results.append((check_name, False))
    
    # Summary
    print_header("📊 Verification Summary")
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    # Final status
    print_header(f"🎯 Overall Status: {'READY TO USE' if all_passed else 'SETUP NEEDED'}")
    
    if all_passed:
        print("🎉 All components are working correctly!")
        print("🚀 You can start using the application immediately.")
    else:
        print("⚠️  Some components need attention.")
        print("👆 Follow the instructions above to fix any issues.")
    
    # Always provide instructions
    provide_instructions()
    
    print(f"\n{'='*60}")
    print("  Verification completed! 🎊")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
