# 🚀 WINDOWS SETUP GUIDE

## Hướng dẫn setup đặc biệt cho Windows

### ⚠️ Unicode Issues trên Windows

Nếu gặp lỗi `UnicodeEncodeError: 'charmap' codec`, thực hiện:

#### Option 1: Set Environment Variables (Khuyến nghị)

```cmd
# Trong Command Prompt
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

# Trong PowerShell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
```

#### Option 2: Run với chcp

```cmd
chcp 65001
python quick_start.py
```

#### Option 3: Manual Step-by-step

```bash
# Bỏ qua automated script, chạy từng bước
python setup_nltk.py
python check_data_format.py
python quick_training_test.py
```

### 🔧 Quick Windows Setup

```cmd
# 1. Set encoding
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers scikit-learn pandas numpy nltk textblob

# 3. Setup NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Check dataset
python check_data_format.py

# 5. Run training test (Windows-safe)
python quick_training_test.py 2>nul
```

### ✅ Windows-Compatible Commands

```cmd
# Health check
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers available')"

# Quick validation
python check_data_format.py
python quick_training_test.py

# Full training
python train_enhanced_100k_fixed.py
```

### 📝 Manual Verification Steps

1. **Check Python**: `python --version` (should be 3.8+)
2. **Check PyTorch**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Check Dataset**: `dir training_data` (should show .json file)
4. **Check NLTK**: `python -c "import nltk; print('NLTK OK')"`

### 🎯 Success Indicators

- ✅ No import errors
- ✅ Dataset file exists
- ✅ Training runs without Unicode errors
- ✅ Model trains successfully

### 📞 Windows Troubleshooting

| Error                | Solution                        |
| -------------------- | ------------------------------- |
| `UnicodeEncodeError` | Set `PYTHONIOENCODING=utf-8`    |
| `CUDA not found`     | Install CUDA toolkit or use CPU |
| `Module not found`   | Reinstall with pip              |
| `Permission denied`  | Run as administrator            |
