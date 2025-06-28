# Commit Analysis Pipeline

Pipeline phân tích commit messages và metadata từ GitHub để hỗ trợ quản lý dự án.

## Tổng quan

Pipeline này phân tích các commit messages và metadata từ Git repositories để cung cấp các thông tin hỗ trợ quản lý dự án như:

- Phân loại loại nhiệm vụ commit (Task Type)
- Dự đoán độ phức tạp (Complexity)
- Xác định lĩnh vực kỹ thuật (Technical Area)
- Xác định kỹ năng cần thiết (Required Skills)
- Dự đoán mức độ ưu tiên (Priority)

Kết quả phân tích và dự đoán sẽ được sử dụng để đưa ra các đề xuất giúp quản lý dự án hiệu quả hơn.

## Kiến trúc

Pipeline bao gồm các thành phần chính sau:

1. **Thu thập dữ liệu** (`data_collection`): Thu thập commit messages và metadata từ GitHub API.
2. **Xử lý dữ liệu** (`data_processing`):
   - Trích xuất đặc trưng từ commit messages và metadata
   - Tự động gán nhãn cho dữ liệu
   - Chuẩn bị dữ liệu cho huấn luyện
3. **Mô hình** (`models`):
   - Mô hình fusion đa phương thức kết hợp text và metadata
   - Mô-đun mã hóa text (LSTM hoặc Transformer)
   - Mô-đun mã hóa metadata
   - Mô-đun fusion Multi-head Attention
   - Các đầu dự đoán cho từng nhiệm vụ
4. **Huấn luyện** (`training`): Huấn luyện mô hình với loss đa nhiệm vụ.
5. **Đánh giá** (`evaluation`): Đánh giá mô hình trên tập test và phân tích lỗi.
6. **Dự đoán** (`predictor`): Dự đoán và đưa ra các đề xuất cho commit mới.

## Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch 1.8+
- Các thư viện phụ thuộc (xem `requirements.txt`)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Sử dụng

### Thu thập dữ liệu

```bash
python main.py collect --github_token YOUR_GITHUB_TOKEN --repo_names owner1/repo1 owner2/repo2 --max_commits 1000
```

### Xử lý dữ liệu

```bash
python main.py process --input_file path/to/raw_data.json
```

### Huấn luyện mô hình

```bash
python main.py train --train_path path/to/train.json --val_path path/to/val.json --num_epochs 50 --text_encoder transformer
```

### Đánh giá mô hình

```bash
python main.py evaluate --test_path path/to/test.json --model_path path/to/model_checkpoint.pt
```

### Dự đoán cho một commit với metadata

```bash
python main.py predict --model_path path/to/model_checkpoint.pt \
    --commit_message "fix: resolve authentication bug in login module" \
    --metadata_file path/to/metadata.json
```

Ví dụ metadata.json:

```json
{
  "author": "developer1",
  "files_changed": 3,
  "additions": 25,
  "deletions": 10,
  "repository": "myproject/backend"
}
```

### Dự đoán cho một batch commit

```bash
python main.py predict_batch --model_path path/to/model_checkpoint.pt --input_file path/to/commits.json
```

### Chạy toàn bộ pipeline end-to-end

```bash
python main.py end_to_end --github_token YOUR_GITHUB_TOKEN --repo_names owner1/repo1 owner2/repo2 --num_epochs 50
```

## Cấu trúc thư mục

```
commit_analysis_pipeline/
├── __init__.py
├── main.py                      # Entry point
├── pipeline.py                  # Integration module
├── data_collection/             # Data collection module
│   ├── __init__.py
│   └── github_collector.py      # GitHub API collector
├── data_processing/             # Data processing module
│   ├── __init__.py
│   ├── commit_processor.py      # Feature extraction & labeling
│   ├── text_processor.py        # Text tokenization & encoding
│   ├── metadata_processor.py    # Metadata normalization
│   └── dataset.py               # PyTorch Dataset & DataLoader
├── models/                      # Model architecture
│   ├── __init__.py
│   ├── multimodal_fusion_model.py # Multimodal fusion model
│   └── predictor.py             # Prediction & recommendation
├── training/                    # Training module
│   ├── __init__.py
│   └── trainer.py               # Training loop & monitoring
└── evaluation/                  # Evaluation module
    ├── __init__.py
    └── evaluator.py             # Metrics & error analysis
```

## Mô hình

### Kiến trúc mô hình

Mô hình **Enhanced Multimodal Fusion Model** kết hợp:

- **Text Encoder**:
  - LSTM hai chiều (BiLSTM) hoặc Transformer để mã hóa commit messages
  - Hỗ trợ cả hai phương pháp mã hóa text với khả năng chuyển đổi linh hoạt
- **Metadata Encoder**:
  - Multi-layer Perceptron (MLP) để mã hóa metadata số
  - Normalization và feature engineering tự động
- **Fusion Module**:
  - Multi-head Cross-Attention để kết hợp đặc trưng text và metadata
  - Residual connections và layer normalization
- **Task Heads**:
  - Các đầu dự đoán riêng biệt cho từng nhiệm vụ
  - Hỗ trợ cả classification (single-label và multi-label) và regression
  - Shared representation learning across tasks

### Dữ liệu đầu vào

- **Text**: Commit messages được tokenize và encode thành vectors
- **Metadata**: Thông tin định lượng về commit:
  - Thông tin tác giả (author_id)
  - Số lượng file thay đổi (files_changed)
  - Số dòng thêm/xóa (additions, deletions, total_changes)
  - Loại file và thư mục được modify
  - Thông tin repository
  - Các đặc trưng được trích xuất tự động (text_length, word_count, risk_level, v.v.)

### Đầu ra

Mô hình dự đoán đa nhiệm vụ (multi-task) với các task sau:

- **Task Type**: Loại nhiệm vụ commit (7 lớp multi-label)
  - Bug Fix, Feature Addition, Refactoring, Documentation, Testing, Build/CI, Other
- **Complexity**: Độ phức tạp của commit (3 lớp)
  - Simple, Moderate, Complex
- **Technical Area**: Lĩnh vực kỹ thuật (5 lớp multi-label)
  - Frontend, Backend, Database, DevOps, Testing
- **Required Skills**: Kỹ năng cần thiết (10 lớp multi-label)
  - Programming, Testing, Database, DevOps, UI/UX, Security, Performance, Documentation, Architecture, Other
- **Priority**: Mức độ ưu tiên (4 lớp)
  - Low, Medium, High, Critical

### Đặc điểm mô hình

- **Multi-label Classification**: Hỗ trợ dự đoán nhiều nhãn cùng lúc cho các task như Task Type, Technical Area, Required Skills
- **Single-label Classification**: Dự đoán một nhãn duy nhất cho Complexity và Priority
- **Fusion Architecture**: Kết hợp hiệu quả thông tin từ text và metadata
- **End-to-end Training**: Huấn luyện đồng thời tất cả các task với shared representations

## Tính năng nâng cao

### Pipeline tích hợp

- **End-to-end workflow**: Từ thu thập dữ liệu đến dự đoán
- **Automatic labeling**: Tự động gán nhãn dựa trên heuristics
- **Flexible architecture**: Dễ dàng thêm/sửa tasks và features
- **Batch prediction**: Hỗ trợ dự đoán hàng loạt cho nhiều commits

### Đánh giá và phân tích

- **Multi-task evaluation**: Đánh giá đồng thời tất cả các tasks
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1-score cho classification
- **Error analysis**: Phân tích chi tiết các lỗi dự đoán
- **Visualization**: Confusion matrices và biểu đồ phân bố lỗi

### Khả năng mở rộng

- **Modular design**: Các module độc lập, dễ maintain
- **Configurable model**: Cấu hình linh hoạt qua JSON
- **Multi-device support**: Hỗ trợ cả CPU và GPU
- **Scalable processing**: Xử lý hiệu quả với datasets lớn

## Đóng góp

Các đóng góp, báo cáo lỗi và đề xuất cải tiến luôn được chào đón!

## Giấy phép

[MIT License](LICENSE)
