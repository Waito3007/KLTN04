# Multi-Modal Fusion Network - Deployment Guide

_Production Deployment Instructions_

## 🚀 Quick Start

The Multi-Modal Fusion Network system is ready for production deployment. Follow these steps to deploy the system:

### 1. System Validation ✅

```bash
# Verify the system is working
cd d:\Project\KLTN04\backend\ai
python comprehensive_test_fixed.py
```

**Expected Output**: All tests pass with "🚀 System ready for production deployment!"

### 2. Real Data Training

#### Option A: Use Existing GitHub Data

```bash
# If you have github_commits_training_data.json
python -c "
from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from multimodal_fusion.training.multitask_trainer import MultiTaskTrainer
import json

# Load real data
with open('training_data/github_commits_training_data.json', 'r') as f:
    data = json.load(f)

# Initialize and train (implement full training loop)
print('Ready for real data training!')
"
```

#### Option B: Generate More Synthetic Data

```bash
python -c "
from multimodal_fusion.data.synthetic_generator import GitHubDataGenerator

generator = GitHubDataGenerator()
large_dataset = generator.generate_dataset(num_samples=10000)
print(f'Generated {len(large_dataset)} samples for training')
"
```

### 3. Integration with Main Application

#### Add to FastAPI Backend

```python
# In backend/api/routes/ai_routes.py
from backend.ai.multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from backend.ai.multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from backend.ai.multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor

@app.post("/analyze-commit-multimodal")
async def analyze_commit_multimodal(commit_data: dict):
    # Load trained model
    model = MultiModalFusionNetwork.load_pretrained("path/to/trained/model.pth")

    # Process input
    text_processor = TextProcessor.load("path/to/text_processor.pkl")
    metadata_processor = MetadataProcessor.load("path/to/metadata_processor.pkl")

    # Make predictions
    predictions = model.predict(commit_data)

    return {
        "risk_prediction": predictions["risk_prediction"],
        "complexity_prediction": predictions["complexity_prediction"],
        "hotspot_prediction": predictions["hotspot_prediction"],
        "urgency_prediction": predictions["urgency_prediction"]
    }
```

### 4. Model Serving Infrastructure

#### Save Trained Model

```python
# After training
import torch

# Save complete model state
torch.save({
    'model_state_dict': model.state_dict(),
    'text_processor': text_processor,
    'metadata_processor': metadata_processor,
    'task_configs': task_configs,
    'model_config': {
        'vocab_size': vocab_size,
        'text_embed_dim': 128,
        'text_hidden_dim': 64,
        'numerical_dim': 34,
        # ... other config
    }
}, 'multimodal_fusion_model_production.pth')
```

#### Load for Inference

```python
# For serving
checkpoint = torch.load('multimodal_fusion_model_production.pth')
model = MultiModalFusionNetwork(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

text_processor = checkpoint['text_processor']
metadata_processor = checkpoint['metadata_processor']
```

### 5. Performance Monitoring

#### Add Metrics Collection

```python
import time
from collections import defaultdict

class ModelMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def log_prediction(self, input_data, predictions, inference_time):
        self.metrics['inference_times'].append(inference_time)
        self.metrics['predictions'].append(predictions)
        # Log to monitoring system

    def get_stats(self):
        return {
            'avg_inference_time': np.mean(self.metrics['inference_times']),
            'total_predictions': len(self.metrics['predictions']),
            'uptime': time.time() - self.start_time
        }
```

### 6. Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model files
COPY multimodal_fusion/ ./multimodal_fusion/
COPY multimodal_fusion_model_production.pth ./

# Copy API code
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose

```yaml
version: "3.8"

services:
  multimodal-ai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/multimodal_fusion_model_production.pth
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 7. Testing in Production

#### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    # Test model loading
    try:
        # Quick inference test
        test_result = model.quick_test()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

#### Load Testing

```bash
# Use artillery or similar for load testing
npm install -g artillery
artillery quick --count 100 --num 10 http://localhost:8000/analyze-commit-multimodal
```

### 8. Monitoring & Logging

#### Structured Logging

```python
import logging
import json

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("multimodal_fusion")

    def log_prediction(self, commit_id, predictions, confidence_scores):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "commit_id": commit_id,
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "model_version": "1.0.0"
        }
        self.logger.info(json.dumps(log_data))
```

### 9. A/B Testing Framework

#### Gradual Rollout

```python
import random

class ABTestManager:
    def __init__(self, multimodal_ratio=0.1):
        self.multimodal_ratio = multimodal_ratio

    def should_use_multimodal(self, user_id):
        # Deterministic based on user_id
        return hash(user_id) % 100 < (self.multimodal_ratio * 100)

    def route_prediction(self, user_id, commit_data):
        if self.should_use_multimodal(user_id):
            return multimodal_model.predict(commit_data)
        else:
            return legacy_model.predict(commit_data)
```

### 10. Backup & Recovery

#### Model Versioning

```python
# Version management
import shutil
from datetime import datetime

def backup_model(model_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"model_backups/multimodal_fusion_{timestamp}.pth"
    shutil.copy2(model_path, backup_path)
    return backup_path

def rollback_model(backup_path, current_path):
    shutil.copy2(backup_path, current_path)
    # Restart service
```

## 🎯 Production Checklist

- [ ] **System Validation**: Run comprehensive_test_fixed.py ✅
- [ ] **Model Training**: Train on real data
- [ ] **API Integration**: Add to FastAPI routes
- [ ] **Docker Setup**: Create containers
- [ ] **Monitoring**: Set up logging and metrics
- [ ] **Health Checks**: Implement status endpoints
- [ ] **Load Testing**: Verify performance
- [ ] **A/B Testing**: Gradual rollout plan
- [ ] **Backup Strategy**: Model versioning
- [ ] **Documentation**: Update API docs

## 📞 Support

For issues during deployment:

1. **Check System Status**: Run the comprehensive test
2. **Review Logs**: Check training and inference logs
3. **Validate Data**: Ensure input format matches expected structure
4. **Monitor Resources**: CPU/Memory usage during inference
5. **Test Endpoints**: Verify API responses

## 🎉 Success Metrics

**Deployment is successful when:**

- ✅ System validation passes
- ✅ Model serves predictions < 100ms
- ✅ API responds to requests
- ✅ Health checks pass
- ✅ Monitoring shows stable performance
- ✅ A/B tests show positive results

---

_Ready for production deployment! 🚀_
