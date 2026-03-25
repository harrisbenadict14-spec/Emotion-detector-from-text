"""
FastAPI backend for AI Emotion Detector
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.predict import EmotionPredictor

# Initialize FastAPI app
app = FastAPI(
    title="AI Emotion Detector API",
    description="A powerful API for detecting emotions from text using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize emotion predictor with correct paths
predictor = EmotionPredictor(
    model_path="model/core_6_emotions_model.pkl",
    vectorizer_path="model/core_6_emotions_vectorizer.pkl", 
    mappings_path="model/core_6_emotions_mappings.pkl",
    encoder_path="model/core_6_emotions_encoder.pkl"
)

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str
    include_probabilities: Optional[bool] = True

class BatchTextInput(BaseModel):
    inputs: List[TextInput]

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    cleaned_text: Optional[str] = None
    probabilities: Optional[dict] = None
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str

class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_type: Optional[str] = None
    vectorizer: Optional[str] = None
    supported_emotions: Optional[List[str]] = None
    features: Optional[str] = None
    error: Optional[str] = None

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Emotion Detector API",
        "version": "1.0.0",
        "description": "Detect emotions from text using machine learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = predictor.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        api_version="1.0.0"
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    info = predictor.get_model_info()
    
    if not info.get('model_loaded', False):
        return ModelInfoResponse(
            model_loaded=False,
            error=info.get('error', 'Unknown error')
        )
    
    return ModelInfoResponse(
        model_loaded=True,
        model_type=info.get('model_type'),
        vectorizer=info.get('vectorizer'),
        supported_emotions=info.get('supported_emotions'),
        features=str(info.get('features', 'Unknown'))
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(input_data: TextInput):
    """
    Predict emotion for a single text
    
    Args:
        input_data: TextInput containing text and optional probabilities flag
        
    Returns:
        PredictionResponse with emotion prediction and confidence
    """
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = predictor.predict(
        input_data.text,
        include_probabilities=input_data.include_probabilities
    )
    
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])
    
    return PredictionResponse(**result)

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_emotions_batch(input_data: BatchTextInput):
    """
    Predict emotions for multiple texts
    
    Args:
        input_data: BatchTextInput containing list of text inputs
        
    Returns:
        BatchPredictionResponse with list of predictions
    """
    if not input_data.inputs:
        raise HTTPException(status_code=400, detail="Input list cannot be empty")
    
    # Extract texts and options
    texts = [item.text for item in input_data.inputs]
    include_probabilities = input_data.inputs[0].include_probabilities if input_data.inputs else True
    
    # Validate texts
    for i, text in enumerate(texts):
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"Text at index {i} cannot be empty")
    
    # Make predictions
    results = predictor.predict_batch(texts, include_probabilities)
    
    # Convert to response format
    predictions = []
    for result in results:
        if 'error' in result:
            predictions.append(PredictionResponse(**result))
        else:
            predictions.append(PredictionResponse(**result))
    
    return BatchPredictionResponse(predictions=predictions)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🧠 AI Emotion Detector API")
    print("=" * 50)
    
    # Check if model is loaded
    if predictor.model:
        print("✅ Model loaded successfully")
        model_info = predictor.get_model_info()
        print(f"🤖 Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"🎯 Supported Emotions: {', '.join(model_info.get('supported_emotions', []))}")
    else:
        print("❌ Model not loaded. Please train the model first:")
        print("   cd model && python train.py")
    
    print("\n🚀 Starting API server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
