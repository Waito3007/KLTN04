# Hướng dẫn sử dụng Kaggle Dataset cho HAN Model

## 🎯 Tổng quan

Bộ công cụ này giúp bạn tải dataset commit từ Kaggle và train mô hình HAN (Hierarchical Attention Network) một cách tự động.

## 📁 Các file chính

- `setup_kaggle.py`: Script setup và cài đặt dependencies
- `download_kaggle_dataset.py`: Tải và xử lý dataset từ Kaggle
- `train_han_with_kaggle.py`: Script tích hợp tải dữ liệu + training HAN
- `README_kaggle_setup.md`: File hướng dẫn này

## 🚀 Hướng dẫn sử dụng

### Bước 1: Setup ban đầu

```bash
# Chạy script setup để cài đặt dependencies và cấu hình Kaggle API
python setup_kaggle.py
```

Script này sẽ:
- Cài đặt các package cần thiết (kaggle, pandas, numpy, etc.)
- Kiểm tra cấu hình Kaggle API
- Hướng dẫn setup API key nếu cần

### Bước 2: Cấu hình Kaggle API (nếu chưa có)

1. Truy cập: https://www.kaggle.com/settings
2. Scroll xuống phần "API" và click "Create New API Token"
3. Download file `kaggle.json`
4. Đặt file vào thư mục:
   - **Windows**: `C:\Users\<username>\.kaggle\`
   - **Linux/Mac**: `~/.kaggle/`
5. Cấp quyền cho file (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

### Bước 3: Tải và train model

```bash
# Chạy script tích hợp để tải dữ liệu và train HAN model
python train_han_with_kaggle.py
```

## 📋 Các tùy chọn có sẵn

### Option 1: Full Pipeline (Khuyên dùng)
- Tự động tải dataset từ Kaggle
- Xử lý và chuẩn hóa dữ liệu
- Train mô hình HAN với dữ liệu mới

### Option 2: Chỉ tải dữ liệu
```bash
python download_kaggle_dataset.py
```

### Option 3: Train với dữ liệu có sẵn
- Sử dụng dữ liệu đã tải trước đó
- Bỏ qua bước download

## 📊 Datasets phổ biến được hỗ trợ

1. `shashankbansal6/git-commits-message-dataset`
2. `madhav28/git-commit-messages`
3. `aashita/git-commit-messages`
4. `jainaru/commit-classification-dataset`
5. `shubhamjain0594/commit-message-generation`
6. `saurabhshahane/conventional-commit-messages`
7. `devanshunigam/commits`
8. `ashydv/commits-dataset`

## 🔧 Cấu hình chi tiết

### Cấu trúc dữ liệu output

Dữ liệu được xử lý theo format phù hợp với HAN model:

```json
{
  "metadata": {
    "total_samples": 10000,
    "created_at": "2025-06-07T10:30:00",
    "source": "kaggle_datasets",
    "statistics": {
      "commit_type": {"feat": 3000, "fix": 2500, "docs": 1000, ...},
      "purpose": {"Feature Implementation": 3500, "Bug Fix": 2800, ...},
      "sentiment": {"neutral": 6000, "positive": 2500, ...},
      "tech_tag": {"javascript": 2000, "python": 1800, ...}
    }
  },
  "data": [
    {
      "text": "Add user authentication feature with JWT tokens",
      "labels": {
        "commit_type": "feat",
        "purpose": "Feature Implementation",
        "sentiment": "neutral",
        "tech_tag": "javascript",
        "author": "john_doe",
        "source_repo": "myapp"
      }
    }
  ]
}
```

### Các nhãn được phân loại tự động

#### 1. Commit Type
- `feat`: Tính năng mới
- `fix`: Sửa bug
- `docs`: Cập nhật documentation
- `style`: Định dạng code
- `refactor`: Tái cấu trúc code
- `test`: Thêm/sửa test
- `chore`: Công việc bảo trì
- `other`: Khác

#### 2. Purpose
- Feature Implementation
- Bug Fix
- Refactoring
- Documentation Update
- Test Update
- Security Patch
- Code Style/Formatting
- Build/CI/CD Script Update
- Other

#### 3. Sentiment
- `positive`: Tích cực
- `negative`: Tiêu cực
- `neutral`: Trung tính
- `urgent`: Khẩn cấp

#### 4. Tech Tag
- `javascript`, `python`, `java`, `react`, `vue`, `angular`
- `css`, `html`, `database`, `api`, `docker`
- `git`, `testing`, `security`, `performance`, `ui`
- `general`: Tổng quát

## 📈 Model Configuration

HAN model được cấu hình tự động dựa trên kích thước dataset:

| Dataset Size | Batch Size | Epochs | Learning Rate |
|--------------|------------|--------|---------------|
| < 1K samples | 16 | 50 | 0.001 |
| 1K - 10K | 32 | 30 | 0.001 |
| > 10K | 64 | 20 | 0.001 |

## 🗂️ Cấu trúc thư mục

```
backend/ai/
├── kaggle_data/                 # Dữ liệu thô từ Kaggle
├── training_data/               # Dữ liệu đã xử lý
├── models/han_kaggle_model/     # Model đã train
├── training_logs/               # Log quá trình training
├── checkpoints/                 # Checkpoint trong quá trình training
├── setup_kaggle.py
├── download_kaggle_dataset.py
├── train_han_with_kaggle.py
└── README_kaggle_setup.md
```

## 🐛 Troubleshooting

### Lỗi phổ biến

1. **"kaggle.json not found"**
   - Đảm bảo file kaggle.json được đặt đúng thư mục
   - Kiểm tra quyền file (600 trên Linux/Mac)

2. **"Dataset not found"**
   - Kiểm tra tên dataset có đúng format `username/dataset-name`
   - Thử với dataset khác trong danh sách phổ biến

3. **"Out of memory during training"**
   - Giảm batch_size trong cấu hình
   - Giảm kích thước dataset hoặc dùng sampling

4. **"No CSV files found"**
   - Một số dataset có format khác (.json, .txt)
   - Kiểm tra thủ công trong thư mục `kaggle_data/`

### Logs và Debugging

- Training logs: `training_logs/training_log_YYYYMMDD_HHMMSS.txt`
- Error logs: Console output với timestamp
- Model checkpoints: `checkpoints/` directory

## 💡 Tips

1. **Chọn dataset phù hợp**: Ưu tiên các dataset có nhiều samples và quality tốt
2. **Preprocessing**: Script tự động làm sạch dữ liệu nhưng có thể cần fine-tune thêm
3. **Memory management**: Với dataset lớn, cân nhắc dùng sampling hoặc streaming
4. **Validation**: Luôn kiểm tra chất lượng dữ liệu trước khi training

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra console logs để xác định lỗi cụ thể
2. Đảm bảo đã setup đúng Kaggle API
3. Thử với dataset nhỏ hơn để test

## 🔄 Cập nhật

Để cập nhật dataset hoặc retrain model:
```bash
# Tải lại với force download
python download_kaggle_dataset.py
# Chọn option force download khi được hỏi

# Hoặc chạy full pipeline với dữ liệu mới
python train_han_with_kaggle.py
```
