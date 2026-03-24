# 📖 AI Emotion Detector - User Manual

## 🚀 Quick Start Guide

### System Requirements
- **Python 3.8+** installed
- **Web browser** (Chrome, Firefox, Safari, Edge)
- **Internet connection** (for initial package installation)

---

## 📋 Step-by-Step Setup

### Step 1: Open Command Prompt/Terminal

**Windows:**
- Press `Win + R`, type `cmd`, press Enter
- Or search for "Command Prompt" in Start Menu

**Mac/Linux:**
- Open Terminal application
- Or press `Ctrl + Alt + T` (Linux)

### Step 2: Navigate to Project Directory

```bash
# Navigate to the project folder
cd "c:\Users\harri\Desktop\Git\emotion detector from text"
```

**Verify you're in the right place:**
```bash
# You should see these folders:
dir  # Windows
ls   # Mac/Linux
```

You should see:
- `backend/`
- `frontend/`
- `dataset/`
- `README.md`
- `start.bat` (Windows) or `start.sh` (Mac/Linux)

---

## 🎯 Option 1: Automatic Startup (Recommended)

### For Windows Users:
```bash
# Double-click this file in File Explorer
start.bat
```

### For Mac/Linux Users:
```bash
# Make executable and run
chmod +x start.sh
./start.sh
```

**This will automatically:**
1. Install all required packages
2. Train the ML model
3. Start the backend server
4. Show you when it's ready

---

## 🔧 Option 2: Manual Setup (Advanced)

### Step 3: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

**What this installs:**
- `fastapi` - Web framework
- `uvicorn` - Server
- `scikit-learn` - Machine learning
- `nltk` - Text processing
- `pandas` - Data handling
- And other required packages

### Step 4: Train the ML Model
```bash
cd model
python train.py
```

**Expected output:**
```
Dataset loaded successfully!
Total samples: 161
Emotion distribution:
happy       30
sad         29
angry       27
fear        26
neutral     25
surprise    24

Training completed!
Model: naive_bayes
Accuracy: 0.4848
Model saved to model.pkl
Vectorizer saved to vectorizer.pkl
```

### Step 5: Start the Backend Server
```bash
cd ..
python main.py
```

**Expected output:**
```
✅ Model loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**IMPORTANT:** Leave this window open! The backend must keep running.

---

## 🌐 Step 6: Open the Frontend

### Method 1: Double-click (Easiest)
1. Open File Explorer
2. Navigate to: `c:\Users\harri\Desktop\Git\emotion detector from text\frontend\`
3. Double-click `index.html`

### Method 2: Browser File Menu
1. Open your web browser
2. Press `Ctrl + O` (Open file)
3. Navigate to: `c:\Users\harri\Desktop\Git\emotion detector from text\frontend\index.html`
4. Click Open

### Method 3: Drag and Drop
1. Open your web browser
2. Drag `index.html` from File Explorer into the browser window

---

## ✅ Step 7: Test the Application

### Test 1: Check Backend Status
Open this URL in your browser:
```
http://localhost:8000
```

You should see:
```json
{
  "message": "Emotion Detection API",
  "version": "1.0.0",
  "status": "active"
}
```

### Test 2: Use the Web Interface
1. In the web app, type: `"I am so happy today!"`
2. Click "Analyze Emotion"
3. You should see:
   - Emotion: **Happy**
   - Confidence: ~35-40%
   - Probability chart with all emotions

### Test 3: Try Different Emotions
Test these examples:
- `"I feel really sad and depressed"` → Should detect **Sad**
- `"This makes me so angry"` → Should detect **Angry**  
- `"I'm scared about what might happen"` → Should detect **Fear**
- `"I'm surprised by this news"` → Should detect **Surprise**
- `"I feel neutral about this"` → Should detect **Neutral**

---

## 🛠️ Troubleshooting Guide

### Problem: "Python not found"
**Solution:**
1. Download Python from https://python.org
2. During installation, check "Add Python to PATH"
3. Restart Command Prompt

### Problem: "pip not found"
**Solution:**
```bash
python -m pip install --upgrade pip
```

### Problem: "Model not loaded" error
**Solution:**
1. Make sure you ran `python train.py` first
2. Check that `model.pkl` exists in `backend/model/` folder
3. Re-run the training script

### Problem: "Connection refused" in browser
**Solution:**
1. Make sure backend server is running (see Step 5)
2. Check the server window shows "Uvicorn running on http://0.0.0.0:8000"
3. Try refreshing the page

### Problem: Frontend not loading
**Solution:**
1. Use a modern web browser (Chrome, Firefox, Safari, Edge)
2. Don't use Internet Explorer
3. Try opening `index.html` directly in browser

### Problem: Low accuracy results
**This is normal!** The model has ~48% accuracy due to:
- Small training dataset (161 samples)
- Simple Naive Bayes algorithm
- Text preprocessing complexity

**To improve accuracy:**
- Add more examples to `dataset/emotions_expanded.csv`
- Retrain the model with `python train.py`

---

## 🎮 How to Use the Application

### Basic Usage
1. **Type your text** in the input box (max 500 characters)
2. **Click "Analyze Emotion"** or press `Ctrl + Enter`
3. **View results** with emotion, confidence, and probability chart
4. **Save result** to history (optional)
5. **Export results** as CSV (optional)

### Advanced Features

#### History Management
- **View History**: Scroll down to see past analyses
- **Clear History**: Click "Clear History" button
- **Automatic Save**: Results are saved automatically

#### Export Functionality
1. Analyze some texts
2. Click "Export" button
3. CSV file downloads with:
   - Timestamp
   - Original text
   - Predicted emotion
   - Confidence score
   - All probability values

#### Statistics Dashboard
- **Total Analyses**: Number of texts analyzed
- **Most Common Emotion**: Your most frequent emotion
- **Average Confidence**: Average confidence across all analyses

#### Theme Toggle
- Click the moon/sun icon in top-right
- Switch between dark and light themes
- Preference is saved automatically

---

## 📱 Mobile Usage

### On Phone/Tablet
1. Transfer the entire project folder to your device
2. Install Python IDE app (like Pydroid 3 for Android)
3. Follow the same steps
4. Open `index.html` in your mobile browser

### Alternative: Web Version
If local setup is difficult, you can:
1. Deploy to a cloud service (Heroku, Vercel, etc.)
2. Access from any device with internet

---

## 🔄 Daily Usage Workflow

### Typical Session
1. **Start backend**: Run `start.bat` or manual commands
2. **Open frontend**: Double-click `index.html`
3. **Analyze texts**: Type and analyze multiple emotions
4. **Review results**: Check history and statistics
5. **Export data**: Download CSV if needed
6. **Close**: Close browser window, stop backend (`Ctrl + C`)

### Pro Tips
- **Batch analysis**: Copy-paste multiple texts
- **Character limit**: Stay under 500 characters for best results
- **Clear input**: Use "Clear" button between analyses
- **Save frequently**: Click "Save Result" for important analyses

---

## 📊 Understanding Results

### Confidence Scores
- **30-50%**: Normal range (model uncertainty)
- **50-70%**: Good confidence
- **70%+**: High confidence (rare with current model)

### Probability Distribution
- Shows likelihood for all 6 emotions
- Should sum to 100%
- Highest probability = predicted emotion

### Model Limitations
- **Context matters**: Same text can have different emotions in different contexts
- **Sarcasm detection**: Model may not understand sarcasm
- **Complex emotions**: Only detects 6 basic emotions
- **Language support**: Only English text supported

---

## 🆘 Getting Help

### Common Questions
**Q: Why is accuracy only ~48%?**
A: This is normal for the current dataset size. Real-world emotion detection is challenging.

**Q: Can I add more training data?**
A: Yes! Edit `dataset/emotions_expanded.csv` and re-run `python train.py`

**Q: Can I use this commercially?**
A: Yes, it's MIT licensed. Just keep the attribution.

**Q: How do I improve the model?**
A: Add more diverse training examples, try different algorithms, or use larger datasets.

### Error Messages Reference
- `"Model not loaded"` → Run training script first
- `"Connection refused"` → Backend server not running
- `"Empty text"` → Type something in the input box
- `"Text too long"` → Stay under 500 characters

### Support Resources
- **README.md**: Technical documentation
- **demo.py**: API testing script
- **Code comments**: Inline documentation
- **FastAPI docs**: Visit `http://localhost:8000/docs`

---

## 🎯 Next Steps

### For Beginners
1. Get comfortable with basic usage
2. Try different types of text
3. Explore the history and export features

### For Developers
1. Read the technical README
2. Examine the API endpoints
3. Modify the training data
4. Experiment with different models

### For Researchers
1. Expand the dataset
2. Try advanced NLP techniques
3. Compare different algorithms
4. Publish your improvements

---

## 📝 Quick Reference

### Essential Commands
```bash
# Start everything (Windows)
start.bat

# Start everything (Mac/Linux)  
./start.sh

# Manual start
cd backend
pip install -r requirements.txt
cd model && python train.py && cd ..
python main.py
```

### Important Files
- `start.bat` / `start.sh` - Quick startup
- `frontend/index.html` - Web interface
- `backend/main.py` - API server
- `backend/model/train.py` - Model training
- `demo.py` - API testing

### URLs
- Frontend: `file:///path/to/frontend/index.html`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

---

## 🎉 You're Ready!

**Congratulations!** You now have a fully functional AI Emotion Detector running locally.

**Remember:**
1. Keep the backend server running while using the app
2. The frontend works in any modern web browser
3. Results are saved in browser's local storage
4. Have fun exploring emotion detection!

**Need help?** Check the troubleshooting section or refer to the comprehensive README.md file.

**Enjoy your AI Emotion Detector! 🧠✨**
