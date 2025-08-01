# AI Models Analysis - TaskFlowAI System

## 🧠 Core AI Architecture

### 1. HAN (Hierarchical Attention Network)

**File**: `han_multitask.pth`
**Location**: `/backend/ai/train_han_multitask.py`

```python
# Hierarchical Attention Network Features:
- Multi-task learning architecture
- Word-level and sentence-level attention
- Purpose classification (9 categories):
  * Feature Implementation
  * Bug Fix
  * Refactoring
  * Documentation Update
  * Test Update
  * Security Patch
  * Code Style/Formatting
  * Build/CI/CD Script Update
  * Other

- Sentiment analysis (3 categories):
  * Positive, Neutral, Negative

- Commit type detection (8 categories):
  * feat, fix, docs, refactor, style, test, chore, uncategorized
```

### 2. CodeBERT Embeddings

**Model**: `microsoft/codebert-base`
**Location**: `/backend/ai/data_preprocessing/embedding_loader.py`

```python
# CodeBERT Integration:
- Pre-trained transformer model
- 768-dimensional embeddings
- Code-aware language understanding
- Optimized for programming text
- Cached embedding computation
- Word and document-level representations
```

### 3. Multi-task Learning Framework

**Location**: `/backend/ai/training/multitask_trainer.py`

```python
# Multi-task Components:
- Uncertainty weighting loss
- Shared representation learning
- Task-specific heads
- Balanced training across tasks
- Performance metrics per task
```

## 🔄 Data Flow Architecture

```
GitHub Commits → Text Preprocessing → CodeBERT Embeddings → HAN Model → Multi-task Outputs
                                                                    ↓
                                                    [Purpose, Sentiment, Type]
                                                                    ↓
                                                          Task Assignment AI
```

## 📊 Training Data Format

### HAN Training Samples

**File**: `han_training_samples.json`

```json
{
  "commit_message": "fix: resolve critical security vulnerability in auth",
  "purpose": "Security Patch",
  "sentiment": "neutral",
  "commit_type": "fix",
  "source_info": {
    "author_name": "developer",
    "repo": "project/repo"
  }
}
```

## 🏗️ Model Architecture Details

### HAN Network Structure

```
Input Text
    ↓
Word Embeddings (CodeBERT)
    ↓
Word-level Attention
    ↓
Sentence Representations
    ↓
Sentence-level Attention
    ↓
Document Representation
    ↓
Multi-task Heads
    ↓
[Purpose, Sentiment, Type] Outputs
```

### Legacy Models (Still Available)

```python
# Backup/Alternative Models:
- commit_classifier_v1.joblib (XGBoost-based)
- commit_classifier.joblib (RandomForest-based)
- Traditional ML with TF-IDF features
```

## 🎯 Use Cases Integration

### For Team Leaders:

- **UC17**: Gợi ý phân công thông minh dựa trên HAN analysis
- **UC18**: Cảnh báo workload từ commit patterns
- **UC16**: AI insights từ multi-task predictions

### For Project Managers:

- **UC19**: Dự đoán tiến độ từ commit analysis
- **UC20**: Báo cáo commit patterns và trends
- **UC08**: Phân tích commit chất lượng

### For Developers:

- **UC08**: Phân tích commit cá nhân
- **UC09**: Phân loại commit messages tự động

## 🔧 Technical Implementation

### Training Pipeline

```bash
# HAN Training Commands:
python train_han_with_kaggle.py      # Train with Kaggle data
python train_han_multitask.py        # Multi-task training
python train_han_github.py           # GitHub-specific training
```

### Model Loading

```python
# In production:
from services.model_loader import ModelLoader
model = ModelLoader.get_instance()
predictions = model.predict(commit_message)
```

### Embedding Preprocessing

```python
# CodeBERT embeddings:
embed_loader = EmbeddingLoader(embedding_type='codebert')
embeddings = embed_loader.get_embeddings_for_doc(document)
```

## 📈 Performance Metrics

### Multi-task Performance:

- **Purpose Classification**: Accuracy tracking per category
- **Sentiment Analysis**: F1-score for balanced classes
- **Commit Type**: Precision/Recall for conventional commits
- **Overall**: Weighted multi-task loss optimization

### Real-time Inference:

- **Embedding Cache**: Pre-computed for common words
- **Model Loading**: Singleton pattern for efficiency
- **Batch Processing**: Support for multiple commits

---

_Detailed AI models analysis for TaskFlowAI system - HAN + CodeBERT architecture_
