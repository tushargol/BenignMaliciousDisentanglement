"""
Autoencoder Service for Industrial IDS
Stage 1: Anomaly Detection using Reconstruction Error
"""
import os
import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import prometheus_client as prom

# Import our model
import sys
sys.path.append('/app')
from src.models.autoencoder import Autoencoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = prom.Counter('autoencoder_requests_total', 'Total requests to autoencoder')
REQUEST_LATENCY = prom.Histogram('autoencoder_request_latency_seconds', 'Request latency')
PREDICTION_COUNT = prom.Counter('autoencoder_predictions_total', 'Total predictions made')
ANOMALY_COUNT = prom.Counter('autoencoder_anomalies_total', 'Total anomalies detected')

class FeatureVector(BaseModel):
    features: List[float]
    timestamp: str = None
    window_id: str = None

class PredictionResponse(BaseModel):
    reconstruction_error: float
    is_anomaly: bool
    confidence: float
    threshold: float
    processing_time_ms: float
    timestamp: str

class AutoencoderService:
    def __init__(self):
        self.model = None
        self.threshold = 0.0
        self.input_dim = int(os.getenv('INPUT_DIM', '196'))
        self.hidden_dim = int(os.getenv('HIDDEN_DIM', '256'))
        self.latent_dim = int(os.getenv('LATENT_DIM', '64'))
        self.threshold_percentile = float(os.getenv('THRESHOLD_PERCENTILE', '75'))
        self.model_path = os.getenv('MODEL_PATH', '/models/autoencoder.pt')
        
        # Load model and threshold
        self.load_model()
        self.load_threshold()
    
    def load_model(self):
        """Load the trained autoencoder model"""
        try:
            self.model = Autoencoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim
            )
            
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                self.model.eval()
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_threshold(self):
        """Load anomaly threshold from environment or calculate from training data"""
        # In production, this should be loaded from training results
        # For now, use environment variable or default
        self.threshold = float(os.getenv('ANOMALY_THRESHOLD', '0.05'))
        logger.info(f"Anomaly threshold set to: {self.threshold}")
    
    @REQUEST_LATENCY.time()
    async def predict(self, features: List[float]) -> Dict[str, Any]:
        """Get reconstruction error for anomaly detection"""
        try:
            start_time = datetime.now()
            
            # Validate input
            if len(features) != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {len(features)}")
            
            # Convert to tensor
            with torch.no_grad():
                x = torch.tensor(features).float().unsqueeze(0)
                recon, _ = self.model(x)
                error = ((x - recon) ** 2).mean().item()
            
            # Determine if anomaly
            is_anomaly = error > self.threshold
            confidence = error / self.threshold if self.threshold > 0 else 0.0
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update metrics
            PREDICTION_COUNT.inc()
            if is_anomaly:
                ANOMALY_COUNT.inc()
            
            return {
                'reconstruction_error': error,
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'threshold': self.threshold,
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize service
service = AutoencoderService()

# FastAPI app
app = FastAPI(
    title="Industrial IDS - Autoencoder Service",
    description="Anomaly detection using autoencoder reconstruction error",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return prom.generate_latest()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": service.model is not None,
        "threshold": service.threshold,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/info")
async def get_info():
    """Get service information"""
    return {
        "service": "autoencoder",
        "version": "1.0.0",
        "model_config": {
            "input_dim": service.input_dim,
            "hidden_dim": service.hidden_dim,
            "latent_dim": service.latent_dim
        },
        "threshold": service.threshold,
        "threshold_percentile": service.threshold_percentile
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(data: FeatureVector, background_tasks: BackgroundTasks):
    """Predict if input features represent an anomaly"""
    REQUEST_COUNT.inc()
    
    try:
        result = await service.predict(data.features)
        
        # Log prediction asynchronously
        background_tasks.add_task(
            log_prediction,
            data.dict(),
            result
        )
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch_predict")
async def batch_predict(data: List[FeatureVector]):
    """Batch prediction for multiple feature vectors"""
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    results = []
    for item in data:
        try:
            result = await service.predict(item.features)
            results.append({
                "window_id": item.window_id,
                **result
            })
        except Exception as e:
            results.append({
                "window_id": item.window_id,
                "error": str(e)
            })
    
    return {"results": results}

async def log_prediction(input_data: Dict, prediction_result: Dict):
    """Log prediction for monitoring"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": prediction_result
    }
    logger.info(f"Prediction logged: {log_entry}")

if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8083,
        reload=False,
        workers=1,
        log_level="info"
    )
