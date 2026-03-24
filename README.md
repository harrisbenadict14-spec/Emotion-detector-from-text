# 🧠 AI Emotion Detector

A professional, production-ready web application that detects emotions from user text input using advanced NLP and Machine Learning techniques.

## ✨ Features

### 🎯 Core Functionality
- **Real-time Emotion Detection**: Analyze text and detect 6 emotions (Happy, Sad, Angry, Fear, Surprise, Neutral)
- **Confidence Scoring**: Get confidence percentages for each prediction
- **Probability Distribution**: View detailed probability breakdowns for all emotions
- **Advanced NLP Pipeline**: Full text preprocessing with tokenization, lemmatization, and stopword removal

### 🎨 Stunning UI/UX
- **Glassmorphism Design**: Modern glass-morphic interface with blur effects
- **Gradient Backgrounds**: Beautiful animated gradient backgrounds
- **Particle Effects**: Floating particles for visual appeal
- **Smooth Animations**: Typing effects, transitions, and micro-interactions
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile

### 🚀 Advanced Features
- **Analysis History**: Track your emotion analysis history
- **Export Results**: Download analysis results as CSV
- **Statistics Dashboard**: View analytics and insights
- **Dark/Light Theme**: Toggle between themes
- **Real-time Character Count**: Track input length
- **Batch Processing**: Analyze multiple texts at once

### 🔧 Technical Features
- **FastAPI Backend**: High-performance async API
- **Machine Learning Model**: Naive Bayes classifier with TF-IDF vectorization
- **RESTful API**: Clean and well-documented endpoints
- **CORS Support**: Ready for frontend integration
- **Error Handling**: Comprehensive error management
- **Local Storage**: Persistent data storage

## 🏗️ Architecture

```
emotion-detector/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── model/
│   │   ├── train.py        # Model training script
│   │   ├── predict.py      # Prediction module
│   │   ├── preprocess.py   # Text preprocessing
│   │   ├── model.pkl       # Trained model
│   │   ├── vectorizer.pkl  # TF-IDF vectorizer
│   │   └── model_mappings.pkl # Label mappings
│   ├── requirements.txt    # Python dependencies
│   └── utils/              # Utility functions
├── frontend/
│   ├── index.html          # Main HTML file
│   ├── style.css           # Styling with glassmorphism
│   └── script.js           # JavaScript functionality
├── dataset/
│   ├── emotions.csv        # Basic dataset
│   └── emotions_expanded.csv # Expanded dataset
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js (optional, for advanced frontend development)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd emotion-detector
```

### 2. Set Up Backend

#### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Train the Model
```bash
cd model
python train.py
```

#### Start the Backend Server
```bash
cd ..
python main.py
```

The backend will start on `http://localhost:8000`

### 3. Open Frontend
Simply open `frontend/index.html` in your web browser or serve it with a web server:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 3000

# Or using Node.js serve
npx serve frontend
```

Visit `http://localhost:3000` to access the application.

## 📋 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Root Endpoint
```http
GET /
```
Returns API information and status.

#### 2. Health Check
```http
GET /health
```
Returns the health status of the API and model.

#### 3. Predict Emotion
```http
POST /predict
Content-Type: application/json

{
  "text": "I am so happy today!",
  "include_probabilities": true
}
```

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.92,
  "probabilities": {
    "happy": 0.92,
    "sad": 0.02,
    "angry": 0.01,
    "fear": 0.02,
    "surprise": 0.02,
    "neutral": 0.01
  },
  "cleaned_text": "happy today"
}
```

#### 4. Batch Prediction
```http
POST /predict-batch
Content-Type: application/json

[
  {
    "text": "I am so happy today!",
    "include_probabilities": true
  },
  {
    "text": "I feel sad",
    "include_probabilities": true
  }
]
```

#### 5. Model Information
```http
GET /model-info
```
Returns information about the loaded model and preprocessing steps.

## 🤖 Machine Learning Model

### Dataset
- **Emotions**: Happy, Sad, Angry, Fear, Surprise, Neutral
- **Samples**: 161 training examples
- **Format**: CSV with text and emotion columns

### Preprocessing Pipeline
1. **Lowercasing**: Convert text to lowercase
2. **URL Removal**: Remove hyperlinks
3. **HTML Tag Removal**: Remove HTML tags
4. **Punctuation Removal**: Remove punctuation and numbers
5. **Tokenization**: Split text into tokens
6. **Stopword Removal**: Remove common English stopwords
7. **Lemmatization**: Reduce words to base form

### Model Details
- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 5000 most frequent n-grams (1-2 grams)
- **Accuracy**: ~48% (can be improved with more data)

### Performance Metrics
```
Accuracy: 0.48
Precision: 0.55 (macro avg)
Recall: 0.48 (macro avg)
F1-Score: 0.45 (macro avg)
```

## 🎨 Frontend Features

### UI Components
- **Hero Section**: Animated title with typing effect
- **Input Section**: Text area with character counter
- **Results Section**: Emotion display with confidence and probabilities
- **History Section**: Analysis history with local storage
- **Statistics Section**: Analytics dashboard
- **Theme Toggle**: Dark/light mode switcher

### Animations
- **Typing Animation**: Dynamic text in hero section
- **Particle Effects**: Floating background particles
- **Gradient Animation**: Shifting background gradients
- **Slide Animations**: Smooth section transitions
- **Hover Effects**: Interactive button and card animations
- **Loading States**: Spinner animations during API calls

### Responsive Design
- **Desktop**: Full-featured experience
- **Tablet**: Optimized layout for medium screens
- **Mobile**: Touch-friendly interface with adjusted layouts

## 🔧 Configuration

### Environment Variables
You can configure the application using environment variables:

```bash
# Backend configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_PATH=model/model.pkl
export VECTORIZER_PATH=model/vectorizer.pkl

# Frontend configuration
export API_BASE_URL=http://localhost:8000
```

### Model Training Options
In `model/train.py`, you can modify:
- `model_type`: Choose between 'naive_bayes' or 'logistic_regression'
- `test_size`: Adjust train/test split (default: 0.2)
- `max_features`: Number of TF-IDF features (default: 5000)
- `ngram_range`: N-gram range (default: (1, 2))

## 📊 Data Management

### Adding More Training Data
To improve model accuracy, add more examples to `dataset/emotions_expanded.csv`:

```csv
text,emotion
"Your text here",emotion_name
```

### Exporting Analysis Results
Click the "Export" button in the results section to download a CSV file with:
- Timestamp
- Original text
- Predicted emotion
- Confidence score
- Probability distribution for all emotions

### Clearing History
- **Individual Results**: Use the "Clear" button
- **All History**: Use "Clear History" in the history section

## 🚀 Deployment

### Backend Deployment (Production)

#### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .
COPY dataset/ ../dataset/

EXPOSE 8000
CMD ["python", "main.py"]
```

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Frontend Deployment
The frontend is static and can be deployed to:
- Netlify
- Vercel
- GitHub Pages
- Any static hosting service

Make sure to update the `API_BASE_URL` in `frontend/script.js` to your production backend URL.

## 🛠️ Development

### Running Tests
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests (if using a test framework)
cd frontend
npm test
```

### Code Style
- **Backend**: Follow PEP 8 Python style guide
- **Frontend**: Use modern JavaScript ES6+ features
- **CSS**: Organized with BEM methodology

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔍 Troubleshooting

### Common Issues

#### Model Loading Error
```
Error: Model not found
```
**Solution**: Make sure to train the model first by running `python model/train.py`

#### CORS Issues
```
CORS policy error
```
**Solution**: The backend has CORS enabled for all origins. For production, update the CORS settings to your specific frontend domain.

#### Low Accuracy
**Solution**: 
- Add more training data to `dataset/emotions_expanded.csv`
- Try different model types (logistic regression)
- Adjust TF-IDF parameters

#### NLTK Download Issues
```
Resource 'punkt_tab' not found
```
**Solution**: The preprocessing script automatically downloads required NLTK data. If issues persist, manually download:
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Performance Optimization
- **Backend**: Use Redis for caching predictions
- **Frontend**: Implement virtual scrolling for large history
- **Model**: Use more advanced algorithms like BERT for better accuracy

## 📈 Future Enhancements

### Planned Features
- [ ] **Advanced Models**: Integration with transformer models (BERT, RoBERTa)
- [ ] **Multi-language Support**: Support for languages other than English
- [ ] **Sentiment Analysis**: Add sentiment scoring alongside emotion detection
- [ ] **Real-time Analysis**: WebSocket support for live text analysis
- [ ] **User Authentication**: Save analysis history per user
- [ ] **API Rate Limiting**: Prevent abuse with rate limiting
- [ ] **Model Monitoring**: Track model performance over time
- [ ] **A/B Testing**: Test different models and preprocessing techniques

### Model Improvements
- [ ] **Larger Dataset**: Collect and label more training data
- [ ] **Data Augmentation**: Generate synthetic training examples
- [ ] **Ensemble Methods**: Combine multiple models for better accuracy
- [ ] **Fine-tuning**: Fine-tune pre-trained language models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FastAPI** - Modern, fast web framework for building APIs
- **Scikit-learn** - Machine learning library for Python
- **NLTK** - Natural Language Toolkit
- **Font Awesome** - Icon library
- **Google Fonts** - Inter font family

## 📞 Support

For support, please:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include error messages and steps to reproduce

---

**Built with ❤️ using Python, FastAPI, and modern web technologies**
