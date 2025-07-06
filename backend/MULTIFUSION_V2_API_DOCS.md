# MultiFusion V2 API Documentation

## Overview

MultiFusion V2 là phiên bản nâng cấp của hệ thống AI phân tích commit, sử dụng architecture BERT + MLP Fusion để cung cấp khả năng phân tích commit chính xác và insights sâu sắc về thành viên team.

## Features

### 🤖 Advanced AI Model
- **Architecture**: BERT (DistilBERT) + Multi-Layer Perceptron Fusion
- **Multi-modal Analysis**: Text + Numerical + Categorical features
- **High Accuracy**: Precision cao trong phân loại commit type
- **Confidence Scoring**: Cung cấp độ tin cậy cho mỗi prediction

### 📊 Comprehensive Analysis
- **Commit Type Classification**: 12 loại commit types
- **Programming Language Detection**: 16+ ngôn ngữ lập trình
- **Code Metrics Integration**: Lines added/removed, files count
- **Member Profiling**: Phân tích style và characteristics của developer

### 🔧 Supported Commit Types
- `feat`: New features
- `fix`: Bug fixes  
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks
- `build`: Build system changes
- `ci`: CI/CD changes
- `perf`: Performance improvements
- `revert`: Reverting changes
- `other_type`: Other types

## API Endpoints

### 1. Model Status Check
```
GET /api/repositories/{repo_id}/ai/model-v2-status
```
Kiểm tra trạng thái và thông tin của MultiFusion V2 model.

**Response Example:**
```json
{
  "success": true,
  "repository_id": 1,
  "model_info": {
    "model_name": "MultiFusion V2",
    "architecture": "BERT + MLP Fusion",
    "version": "2.0",
    "is_available": true,
    "device": "cuda",
    "supported_languages": ["python", "javascript", "java", ...],
    "supported_commit_types": ["feat", "fix", "docs", ...]
  }
}
```

### 2. Single Commit Analysis
```
POST /api/repositories/analyze-multifusion-v2-commit
```
Phân tích một commit đơn lẻ với đầy đủ features.

**Request Body:**
```json
{
  "commit_message": "feat: implement user authentication with JWT",
  "lines_added": 145,
  "lines_removed": 23,
  "files_count": 8,
  "detected_language": "python"
}
```

**Response Example:**
```json
{
  "success": true,
  "result": {
    "commit_type": "feat",
    "confidence": 0.996,
    "all_probabilities": {
      "feat": 0.996,
      "fix": 0.002,
      "docs": 0.001,
      ...
    },
    "input_features": {
      "lines_added": 145,
      "lines_removed": 23,
      "files_count": 8,
      "detected_language": "python",
      "total_changes": 168,
      "ratio_added_removed": 6.3
    }
  }
}
```

### 3. Member Commits Analysis
```
GET /api/repositories/{repo_id}/members/{member_login}/commits-v2
```
Phân tích tất cả commits của một member với insights chi tiết.

**Query Parameters:**
- `branch_name` (optional): Filter by branch
- `limit` (optional): Maximum commits to analyze (default: 50)

**Response Example:**
```json
{
  "success": true,
  "repository_id": 1,
  "member_login": "john_doe",
  "model_used": "MultiFusion V2",
  "analysis": {
    "total_commits": 25,
    "commit_type_distribution": {
      "feat": 10,
      "fix": 8,
      "docs": 4,
      "refactor": 3
    },
    "dominant_commit_type": {
      "type": "feat",
      "count": 10,
      "percentage": 40.0
    },
    "productivity_metrics": {
      "total_changes": 1250,
      "total_files_modified": 85,
      "avg_changes_per_commit": 50.0,
      "avg_files_per_commit": 3.4
    },
    "languages_used": ["python", "javascript"],
    "commits": [...] // Individual commit analysis
  }
}
```

### 4. Batch Analysis
```
POST /api/repositories/{repo_id}/ai/batch-analyze-v2
```
Phân tích nhiều commits cùng lúc.

**Request Body:**
```json
{
  "commits": [
    {
      "message": "fix: resolve memory leak",
      "lines_added": 45,
      "lines_removed": 12,
      "files_count": 3,
      "detected_language": "python"
    },
    ...
  ]
}
```

### 5. Model Comparison
```
GET /api/repositories/{repo_id}/ai/compare-models
```
So sánh kết quả từ MultiFusion V1 và V2.

**Query Parameters:**
- `commit_message`: Test commit message
- `lines_added`: Lines added
- `lines_removed`: Lines removed
- `files_count`: Files count
- `detected_language`: Programming language

## Member Analysis Insights

### Developer Profiling
MultiFusion V2 có thể tự động phân loại developer types:

- **🚀 FEATURE BUILDER**: Focuses on new functionality (≥40% feat commits)
- **🔧 BUG HUNTER**: Specializes in fixing issues (≥30% fix commits)
- **🛠️ CODE OPTIMIZER**: Focuses on code improvement (≥25% refactor commits)
- **✅ QUALITY ASSURER**: Emphasizes testing (≥20% test commits)
- **📚 DOCUMENTATION CHAMPION**: Focuses on documentation (≥15% docs commits)
- **🎯 BALANCED CONTRIBUTOR**: Well-rounded development

### Commit Style Analysis
- **LARGE COMMITS**: >100 average changes per commit
- **MEDIUM COMMITS**: 50-100 average changes per commit  
- **SMALL COMMITS**: <50 average changes per commit

### Productivity Levels
- **HIGH**: >500 total changes
- **MEDIUM**: 200-500 total changes
- **GROWING**: <200 total changes

## Installation & Setup

### 1. Model Files
Đảm bảo các files sau tồn tại:
```
backend/ai/models/multifusion/
├── multifusionV2.pth         # Trained model weights
├── metadata_v2.json          # Model metadata
└── data/                     # Training data (optional)
```

### 2. Dependencies
```bash
pip install torch transformers scikit-learn numpy
```

### 3. GPU Support (Optional)
Để sử dụng GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage Examples

### Python Service Usage
```python
from services.multifusion_v2_service import MultiFusionV2Service

# Initialize service
service = MultiFusionV2Service()

# Check if model is available
if service.is_model_available():
    # Analyze single commit
    result = service.predict_commit_type(
        "feat: add user authentication",
        lines_added=100,
        lines_removed=10,
        files_count=5,
        detected_language="python"
    )
    print(f"Predicted type: {result['commit_type']}")
    print(f"Confidence: {result['confidence']:.3f}")
```

### API Usage with curl
```bash
# Test single commit
curl -X POST "http://localhost:8000/api/repositories/analyze-multifusion-v2-commit" \
     -H "Content-Type: application/json" \
     -d '{
       "commit_message": "feat: implement user authentication",
       "lines_added": 100,
       "lines_removed": 10,
       "files_count": 5,
       "detected_language": "python"
     }'

# Check model status
curl "http://localhost:8000/api/repositories/1/ai/model-v2-status"
```

## Performance Metrics

- **Accuracy**: ~95% trên validation set
- **Inference Speed**: ~50ms per commit trên GPU
- **Memory Usage**: ~2GB GPU memory
- **Supported Languages**: 16+ programming languages
- **Confidence Threshold**: Recommended ≥0.8 for high-confidence predictions

## Troubleshooting

### Model Not Available
```json
{
  "success": false,
  "error": "Model not available"
}
```
**Solutions:**
1. Kiểm tra file `multifusionV2.pth` có tồn tại
2. Kiểm tra file `metadata_v2.json` 
3. Chạy `python create_metadata.py` để tạo metadata mới

### Out of Memory
**Solutions:**
1. Giảm batch size
2. Sử dụng CPU thay vì GPU
3. Tăng GPU memory

### Unknown Language Error
Model sẽ fallback về "unknown_language" cho các ngôn ngữ không nhận diện được.

## Advanced Features

### Custom Model Training
Để train model với data mới:
```bash
cd backend/ai/models/multifusion/train/
python train_multifusion_v2.py
```

### Metadata Recreation
Để tạo lại metadata từ training data:
```bash
cd backend/ai/models/multifusion/
python create_metadata.py
```
