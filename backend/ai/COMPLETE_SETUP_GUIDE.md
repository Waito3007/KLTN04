# 🚀 HƯỚNG DẪN HOÀN CHỈNH: MULTIMODAL FUSION MODEL

## Hướng dẫn từ A đến Z để setup, tạo dataset và train model

---

## 📋 MỤC LỤC

1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#cài-đặt-môi-trường)
3. [Tạo và chuẩn bị dataset](#tạo-và-chuẩn-bị-dataset)
4. [Kiểm tra hệ thống](#kiểm-tra-hệ-thống)
5. [Training model](#training-model)
6. [Đánh giá model](#đánh-giá-model)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## 🖥️ YÊU CẦU HỆ THỐNG

### Phần cứng

- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB+
- **GPU**: CUDA-compatible (optional, khuyến nghị cho training nhanh)
- **Storage**: Ít nhất 10GB trống
- **CPU**: Multi-core processor

### Phần mềm

- **Python**: 3.8+ (khuyến nghị 3.9-3.11)
- **Git**: Latest version
- **CUDA Toolkit**: 11.8+ (nếu có GPU)

---

## ⚙️ CÀI ĐẶT MÔI TRƯỜNG

### Bước 1: Clone Repository

```bash
# Clone project
git clone <your-repository-url>
cd KLTN04

# Chuyển đến thư mục AI
cd backend/ai
```

### Bước 2: Tạo Virtual Environment

```bash
# Tạo virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux/Mac)
source venv/bin/activate
```

### Bước 3: Cài đặt Dependencies

```bash
# Cài đặt requirements cơ bản
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install scikit-learn
pip install pandas numpy
pip install nltk textblob
pip install jupyter notebook
pip install matplotlib seaborn
pip install tqdm
pip install kaggle

# Hoặc từ file requirements (nếu có)
pip install -r requirements.txt
```

### Bước 4: Setup NLTK

```bash
# Chạy script setup NLTK
python setup_nltk.py
```

Hoặc manual setup:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
```

### Bước 5: Kiểm tra cài đặt

```bash
# Test imports
python test_imports.py
```

---

## 📊 TẠO VÀ CHUẨN BỊ DATASET

### Option 1: Sử dụng Dataset có sẵn (Khuyến nghị)

```bash
# Kiểm tra dataset có sẵn
python check_data_format.py
```

Nếu file `training_data/improved_100k_multimodal_training.json` đã có:

- ✅ Dataset đã sẵn sàng (100K samples)
- ✅ Có thể bỏ qua bước tạo dataset

### Option 2: Tạo Dataset từ GitHub

#### Bước 2.1: Setup Kaggle (nếu cần)

```bash
# Setup Kaggle credentials
python setup_kaggle.py

# Download dataset từ Kaggle
python download_kaggle_dataset.py
```

#### Bước 2.2: Download GitHub Commits

```bash
# Download commits từ GitHub
python download_github_commits.py
```

#### Bước 2.3: Process Dataset

```bash
# Xử lý dataset 100K
python process_large_dataset_100k_v2.py
```

### Option 3: Tạo Simple Dataset để test

```bash
# Tạo dataset nhỏ để test
python simple_dataset_creator.py
```

### Bước 2.4: Kiểm tra Dataset

```bash
# Kiểm tra format và quality của dataset
python check_data_format.py
```

**Expected Output:**

```
Data type: <class 'dict'>
Data keys: ['train_data', 'val_data']
Train samples: 80000, Validation samples: 20000
✅ Dataset format is correct
```

---

## 🔍 KIỂM TRA HỆ THỐNG

### Bước 3.1: Test Components

```bash
# Test individual components
python test_minimal_enhanced_processor.py
python test_multimodal_structure.py
```

### Bước 3.2: Comprehensive System Test

```bash
# Test toàn bộ hệ thống
python evaluate_multimodal_model.py
```

**Expected Output:**

```
🎯 MULTIMODAL MODEL EVALUATION SUMMARY
✅ All tests passed! Model is ready for training.
```

### Bước 3.3: Quick Training Test

```bash
# Test training trên dataset nhỏ
python quick_training_test.py
```

**Expected Output:**

```
🎯 QUICK TRAINING TEST RESULTS
✅ Training Success: True
📊 Training Accuracy: 85%+
📊 Validation Accuracy: 85%+
```

---

## 🚂 TRAINING MODEL

### Bước 4.1: Chọn Training Script

**For Full Training (100K dataset):**

```bash
python train_enhanced_100k_fixed.py
```

**For Quick Test (Small dataset):**

```bash
python quick_training_test.py
```

### Bước 4.2: Monitor Training

Training sẽ hiển thị:

```
2025-06-10 16:29:26,729 - INFO - Epoch 1/50, Batch 0/2500, Loss: 4.5201, LR: 1.00e-03
2025-06-10 16:29:34,515 - INFO - Epoch 1/50, Batch 100/2500, Loss: 1.7963, LR: 1.00e-03
...
```

### Bước 4.3: Training Parameters

```python
# Có thể điều chỉnh trong script:
epochs = 50                    # Số epochs
batch_size = 32               # Batch size
learning_rate = 1e-3          # Learning rate
patience = 10                 # Early stopping patience
```

### Bước 4.4: Model Checkpoints

Models được lưu tại:

```
trained_models/enhanced_multimodal_fusion_100k/
├── best_enhanced_model.pth           # Best model
├── enhanced_training_history.json    # Training history
└── model_config.json                # Model configuration
```

---

## 📈 ĐÁNH GIÁ MODEL

### Bước 5.1: Evaluate Model Performance

```bash
# Đánh giá model đã train
python evaluate_multimodal_fusion.py
```

### Bước 5.2: Generate Final Report

```bash
# Tạo báo cáo cuối cùng
python generate_final_report.py
```

### Bước 5.3: Test Inference

```bash
# Test inference với model đã train
python multimodal_commit_inference.py
```

---

## 🚀 DEPLOYMENT

### Bước 6.1: Model Integration

```bash
# Test integration với backend
python test_multimodal_integration.py
```

### Bước 6.2: API Integration

Integrate model vào backend API:

```python
# Trong backend/services/
from ai.multimodal_fusion.inference import MultimodalInference

# Load trained model
inference = MultimodalInference(
    model_path="ai/trained_models/enhanced_multimodal_fusion_100k/best_enhanced_model.pth"
)

# Predict commit analysis
result = inference.predict(commit_message, metadata)
```

### Bước 6.3: Production Setup

Xem [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) để setup production.

---

## 🛠️ TROUBLESHOOTING

### Lỗi thường gặp và cách khắc phục

#### 1. ImportError: No module named 'torch'

```bash
# Cài lại PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. CUDA out of memory

```bash
# Giảm batch size trong training script
batch_size = 16  # hoặc 8
```

#### 3. NLTK data not found

```bash
# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

#### 4. UnicodeEncodeError

```bash
# Set environment variables
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
```

#### 5. Dataset not found

```bash
# Kiểm tra đường dẫn dataset
ls training_data/
python check_data_format.py
```

#### 6. Model config error

```bash
# Kiểm tra model config
python test_multimodal_structure.py
```

---

## 📚 SCRIPTS REFERENCE

### Core Scripts

| Script                         | Purpose                    | When to use          |
| ------------------------------ | -------------------------- | -------------------- |
| `setup_nltk.py`                | Setup NLTK dependencies    | First time setup     |
| `test_imports.py`              | Test all imports           | After installation   |
| `check_data_format.py`         | Check dataset format       | Before training      |
| `evaluate_multimodal_model.py` | Comprehensive evaluation   | System validation    |
| `quick_training_test.py`       | Quick training test        | Before full training |
| `train_enhanced_100k_fixed.py` | Full model training        | Main training        |
| `generate_final_report.py`     | Generate evaluation report | After training       |

### Data Processing Scripts

| Script                             | Purpose              | Dataset Size |
| ---------------------------------- | -------------------- | ------------ |
| `simple_dataset_creator.py`        | Create test dataset  | ~1K samples  |
| `download_github_commits.py`       | Download from GitHub | Variable     |
| `process_large_dataset_100k_v2.py` | Process 100K dataset | 100K samples |

### Testing Scripts

| Script                               | Purpose              | Test Level   |
| ------------------------------------ | -------------------- | ------------ |
| `test_minimal_enhanced_processor.py` | Test text processor  | Component    |
| `test_multimodal_structure.py`       | Test model structure | Architecture |
| `test_multimodal_integration.py`     | Test integration     | System       |
| `final_system_validation.py`         | Full system test     | Complete     |

---

## ⏱️ TIMELINE ESTIMATE

### Quick Setup (Testing)

- **Setup Environment**: 30 minutes
- **Test Dataset**: 10 minutes
- **Quick Training**: 15 minutes
- **Total**: ~1 hour

### Full Setup (Production)

- **Setup Environment**: 30 minutes
- **Download/Process 100K Dataset**: 2-4 hours
- **Full Training**: 4-8 hours (depending on hardware)
- **Evaluation**: 30 minutes
- **Total**: 7-13 hours

---

## 🎯 SUCCESS CRITERIA

### ✅ Successful Setup Indicators

1. **Environment Setup**:

   ```
   python test_imports.py
   # Output: ✅ All imports successful
   ```

2. **Dataset Ready**:

   ```
   python check_data_format.py
   # Output: ✅ Dataset format is correct
   ```

3. **System Validation**:

   ```
   python evaluate_multimodal_model.py
   # Output: 🎉 All tests passed! Model is ready for training.
   ```

4. **Training Success**:

   ```
   python quick_training_test.py
   # Output: ✅ Training Success: True, Accuracy: 85%+
   ```

5. **Final Evaluation**:
   ```
   python generate_final_report.py
   # Output: 🎉 CONGRATULATIONS! Your multimodal model is production-ready!
   ```

---

## 📞 SUPPORT

### Nếu gặp vấn đề:

1. **Check logs**: Xem file `.log` trong thư mục AI
2. **Run diagnostics**: `python test_core_system.py`
3. **Check system status**: `python evaluate_multimodal_model.py`
4. **Review error messages**: Sử dụng troubleshooting guide

### Common Commands

```bash
# Quick health check
python test_imports.py && python test_multimodal_structure.py

# Full system validation
python evaluate_multimodal_model.py

# Training from scratch
python check_data_format.py && python quick_training_test.py

# Production training
python train_enhanced_100k_fixed.py
```

---

## 🎉 CONCLUSION

Sau khi hoàn thành hướng dẫn này, bạn sẽ có:

- ✅ **Working environment** với tất cả dependencies
- ✅ **100K multimodal dataset** đã được processed
- ✅ **Trained model** với performance 85%+
- ✅ **Production-ready system** để analyze commits
- ✅ **Integration** với backend API

**Next Steps:**

1. Integrate model vào production environment
2. Monitor performance trên real data
3. Fine-tune hyperparameters nếu cần
4. Scale up để handle larger datasets

**Happy Training! 🚀**
