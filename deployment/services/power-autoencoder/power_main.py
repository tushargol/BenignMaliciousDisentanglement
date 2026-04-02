"""
Power Systems Autoencoder Service for Industrial IDS
Stage 1: Anomaly Detection using Reconstruction Error for Power Grid Infrastructure
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
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

# Prometheus metrics for Power Systems
REQUEST_COUNT = prom.Counter('power_autoencoder_requests_total', 'Total requests to power autoencoder')
GRID_REQUEST_COUNT = prom.Counter('power_autoencoder_grid_requests_total', 'Total grid-specific requests')
REQUEST_LATENCY = prom.Histogram('power_autoencoder_request_latency_seconds', 'Request latency')
PREDICTION_COUNT = prom.Counter('power_autoencoder_predictions_total', 'Total predictions made')
ANOMALY_COUNT = prom.Counter('power_autoencoder_anomalies_total', 'Total anomalies detected')
SUBSTATION_ANOMALIES = prom.Counter('power_autoencoder_substation_anomalies_total', 'Substation-specific anomalies', ['substation_id'])
GRID_THREAT_LEVEL = prom.Gauge('power_autoencoder_grid_threat_level', 'Current grid threat level')

class PowerFeatureVector(BaseModel):
    features: List[float]
    timestamp: str = None
    window_id: str = None
    metadata: Optional[Dict[str, Any]] = None

class PowerPredictionResponse(BaseModel):
    reconstruction_error: float
    is_anomaly: bool
    confidence: float
    threshold: float
    processing_time_ms: float
    timestamp: str
    substation_id: Optional[str] = None
    voltage_level: Optional[str] = None
    device_type: Optional[str] = None
    grid_context: Optional[Dict[str, Any]] = None

class PowerAutoencoderService:
    def __init__(self):
        self.model = None
        self.threshold = 0.0
        self.input_dim = int(os.getenv('INPUT_DIM', '196'))
        self.hidden_dim = int(os.getenv('HIDDEN_DIM', '256'))
        self.latent_dim = int(os.getenv('LATENT_DIM', '64'))
        self.threshold_percentile = float(os.getenv('THRESHOLD_PERCENTILE', '75'))
        self.model_path = os.getenv('MODEL_PATH', '/models/power_autoencoder.pt')
        self.substation_id = os.getenv('SUBSTATION_ID', 'SUB001')
        self.power_systems_mode = os.getenv('POWER_SYSTEMS_MODE', 'true').lower() == 'true'
        self.grid_aware_training = os.getenv('GRID_AWARE_TRAINING', 'true').lower() == 'true'
        
        # Load model and threshold
        self.load_model()
        self.load_threshold()
        
        logger.info(f"Power Systems Autoencoder initialized for substation: {self.substation_id}")
        logger.info(f"Power Systems Mode: {self.power_systems_mode}")
        logger.info(f"Grid Aware Training: {self.grid_aware_training}")
    
    def load_model(self):
        """Load the trained power systems autoencoder model"""
        try:
            self.model = Autoencoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim
            )
            
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                self.model.eval()
                logger.info(f"Power Systems model loaded successfully from {self.model_path}")
            else:
                # Fallback to regular model if power systems model not available
                fallback_path = self.model_path.replace('power_', '')
                if os.path.exists(fallback_path):
                    self.model.load_state_dict(torch.load(fallback_path, map_location='cpu'))
                    self.model.eval()
                    logger.warning(f"Using fallback model from {fallback_path}")
                else:
                    logger.error(f"No model file found: {self.model_path} or {fallback_path}")
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load power systems model: {e}")
            raise
    
    def load_threshold(self):
        """Load anomaly threshold from environment or calculate from training data"""
        # In production, this should be loaded from training results
        # For now, use environment variable or default
        self.threshold = float(os.getenv('ANOMALY_THRESHOLD', '0.05'))
        logger.info(f"Power Systems anomaly threshold set to: {self.threshold}")
    
    def extract_power_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract power systems specific metadata"""
        if not metadata:
            return {}
        
        return {
            'substation_id': metadata.get('substation_id', self.substation_id),
            'voltage_level': metadata.get('voltage_level'),
            'device_type': metadata.get('device_type'),
            'bay_id': metadata.get('bay_id'),
            'protection_zone': metadata.get('protection_zone'),
            'grid_state': metadata.get('grid_state', 'normal')
        }
    
    @REQUEST_LATENCY.time()
    async def predict(self, features: List[float], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get reconstruction error for power systems anomaly detection"""
        try:
            start_time = datetime.now()
            
            # Validate input
            if len(features) != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {len(features)}")
            
            # Extract power systems metadata
            power_metadata = self.extract_power_metadata(metadata)
            
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
                SUBSTATION_ANOMALIES.labels(substation_id=power_metadata.get('substation_id', 'unknown')).inc()
                
                # Update grid threat level based on anomaly confidence
                if confidence > 2.0:
                    GRID_THREAT_LEVEL.set(3.0)  # Critical
                elif confidence > 1.5:
                    GRID_THREAT_LEVEL.set(2.0)  # High
                else:
                    GRID_THREAT_LEVEL.set(1.0)  # Medium
            else:
                GRID_THREAT_LEVEL.set(0.0)  # Normal
            
            return {
                'reconstruction_error': error,
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'threshold': self.threshold,
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat(),
                'substation_id': power_metadata.get('substation_id'),
                'voltage_level': power_metadata.get('voltage_level'),
                'device_type': power_metadata.get('device_type'),
                'grid_context': power_metadata
            }
            
        except Exception as e:
            logger.error(f"Power Systems prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize power systems service
service = PowerAutoencoderService()

# FastAPI app
app = FastAPI(
    title="Power Systems IDS - Autoencoder Service",
    description="Anomaly detection for power grid infrastructure using autoencoder reconstruction error",
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
    """Health check endpoint for power systems"""
    return {
        "status": "healthy",
        "model_loaded": service.model is not None,
        "threshold": service.threshold,
        "substation_id": service.substation_id,
        "power_systems_mode": service.power_systems_mode,
        "grid_aware_training": service.grid_aware_training,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/info")
async def get_info():
    """Get power systems service information"""
    return {
        "service": "power-autoencoder",
        "version": "1.0.0",
        "substation_id": service.substation_id,
        "model_config": {
            "input_dim": service.input_dim,
            "hidden_dim": service.hidden_dim,
            "latent_dim": service.latent_dim
        },
        "threshold": service.threshold,
        "threshold_percentile": service.threshold_percentile,
        "power_systems_mode": service.power_systems_mode,
        "grid_aware_training": service.grid_aware_training
    }

@app.get("/metrics/grid")
async def get_grid_metrics():
    """Get power systems specific metrics"""
    return {
        "grid_threat_level": GRID_THREAT_LEVEL._value.get(),
        "total_anomalies": ANOMALY_COUNT._value.get(),
        "substation_anomalies": dict(SUBSTATION_ANOMALIES._value),
        "total_predictions": PREDICTION_COUNT._value.get(),
        "substation_id": service.substation_id
    }

@app.post("/predict", response_model=PowerPredictionResponse)
async def predict_anomaly(data: PowerFeatureVector, background_tasks: BackgroundTasks):
    """Predict if input features represent a power systems anomaly"""
    REQUEST_COUNT.inc()
    GRID_REQUEST_COUNT.inc()
    
    try:
        result = await service.predict(data.features, data.metadata)
        
        # Log prediction asynchronously
        background_tasks.add_task(
            log_power_prediction,
            data.dict(),
            result
        )
        
        return PowerPredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Power Systems prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch_predict")
async def batch_predict(data: List[PowerFeatureVector]):
    """Batch prediction for multiple power systems feature vectors"""
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    results = []
    for item in data:
        try:
            result = await service.predict(item.features, item.metadata)
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

@app.get("/threats/power-systems")
async def get_power_systems_threats():
    """Get power systems specific threat information"""
    return {
        "supported_threats": [
            "relay_trip_attack",
            "scada_command_injection", 
            "pmu_data_manipulation",
            "voltage_stability_attack",
            "frequency_interference",
            "protection_zone_bypass",
            "bay_level_anomaly",
            "circuit_breaker_misoperation"
        ],
        "detection_capabilities": {
            "real_time_detection": True,
            "substation_isolation": True,
            "grid_correlation": service.grid_aware_training,
            "voltage_level_detection": True,
            "device_type_classification": True
        },
        "current_threat_level": GRID_THREAT_LEVEL._value.get()
    }

async def log_power_prediction(input_data: Dict, prediction_result: Dict):
    """Log power systems prediction for monitoring"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "service": "power-autoencoder",
        "substation_id": prediction_result.get('substation_id'),
        "input": input_data,
        "prediction": prediction_result,
        "grid_threat_level": GRID_THREAT_LEVEL._value.get()
    }
    logger.info(f"Power Systems prediction logged: {log_entry}")

if __name__ == "__main__":
    # Run the power systems service
    uvicorn.run(
        "power_main:app",
        host="0.0.0.0",
        port=8083,
        reload=False,
        workers=1,
        log_level="info"
    )
