# 📋 COMMAND REFERENCE CARD

## Quick Commands cho Multimodal Fusion Model

---

## 🚀 QUICK START

```bash
# Automated setup (khuyến nghị cho beginners)
python quick_start.py

# Manual setup từng bước
python setup_nltk.py
python test_imports.py
python check_data_format.py
python evaluate_multimodal_model.py
```

---

## 📊 DATASET COMMANDS

```bash
# Kiểm tra dataset có sẵn
python check_data_format.py

# Tạo test dataset nhỏ
python simple_dataset_creator.py

# Download từ Kaggle
python setup_kaggle.py
python download_kaggle_dataset.py

# Xử lý dataset 100K
python process_large_dataset_100k_v2.py
```

---

## 🧪 TESTING COMMANDS

```bash
# Test imports và dependencies
python test_imports.py

# Test text processor
python test_minimal_enhanced_processor.py

# Test model structure
python test_multimodal_structure.py

# Test toàn bộ hệ thống
python evaluate_multimodal_model.py

# Test training nhanh
python quick_training_test.py
```

---

## 🚂 TRAINING COMMANDS

```bash
# Training nhanh (test)
python quick_training_test.py

# Training đầy đủ 100K
python train_enhanced_100k_fixed.py

# Training với custom parameters
python train_enhanced_100k_fixed.py --epochs 30 --batch_size 16
```

---

## 📈 EVALUATION COMMANDS

```bash
# Đánh giá model
python evaluate_multimodal_fusion.py

# Tạo báo cáo chi tiết
python generate_final_report.py

# Test inference
python multimodal_commit_inference.py

# Validation cuối cùng
python final_system_validation.py
```

---

## 🔧 TROUBLESHOOTING COMMANDS

```bash
# Kiểm tra system status
python test_core_system.py

# Debug imports
python test_imports.py

# Debug text processing
python test_minimal_enhanced_processor.py

# Debug model architecture
python test_multimodal_structure.py

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check dataset
python check_data_format.py
```

---

## 📁 FILE STRUCTURE

```
backend/ai/
├── COMPLETE_SETUP_GUIDE.md          # Hướng dẫn chi tiết
├── quick_start.py                   # Auto setup script
├── train_enhanced_100k_fixed.py     # Main training script
├── evaluate_multimodal_model.py     # System evaluation
├── quick_training_test.py           # Quick training test
└── training_data/
    └── improved_100k_multimodal_training.json
```

---

## ⚡ SHORTCUTS

```bash
# Quick health check
python test_imports.py && python test_multimodal_structure.py

# Full validation
python evaluate_multimodal_model.py

# Training pipeline
python check_data_format.py && python quick_training_test.py

# Production training
python train_enhanced_100k_fixed.py 2>&1 | tee training.log
```

---

## 📊 STATUS INDICATORS

### ✅ Success Indicators

- `✅ All tests passed!`
- `🎉 Training completed successfully!`
- `✅ APPROVED FOR PRODUCTION`

### ❌ Error Indicators

- `❌ Import failed`
- `KeyError: 'embedding_dim'`
- `FileNotFoundError: training_data`

### ⚠️ Warning Indicators

- `⚠️ Not in virtual environment`
- `⚠️ Dataset not found`
- `⚠️ GPU not available`

---

## 🎯 COMMON WORKFLOWS

### First Time Setup

```bash
python quick_start.py
# Follow the automated setup
```

### Development Workflow

```bash
python test_imports.py
python check_data_format.py
python quick_training_test.py
python evaluate_multimodal_model.py
```

### Production Training

```bash
python check_data_format.py
python evaluate_multimodal_model.py
python train_enhanced_100k_fixed.py
python generate_final_report.py
```

### Debugging Issues

```bash
python test_imports.py
python test_multimodal_structure.py
python evaluate_multimodal_model.py
# Check logs for specific errors
```

---

## 📞 HELP

- 📖 **Detailed Guide**: `COMPLETE_SETUP_GUIDE.md`
- 🔍 **System Status**: `python evaluate_multimodal_model.py`
- 📄 **Logs**: Check `.log` files in current directory
- 🆘 **Troubleshooting**: See COMPLETE_SETUP_GUIDE.md section
