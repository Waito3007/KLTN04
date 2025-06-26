# Commit Analysis Pipeline

Pipeline phân tích commit messages và metadata từ GitHub để hỗ trợ quản lý dự án.

## Tổng quan

Pipeline này phân tích các commit messages và metadata từ Git repositories để cung cấp các thông tin hỗ trợ quản lý dự án như:

- Dự đoán mức độ rủi ro (Risk)
- Dự đoán độ phức tạp (Complexity)
- Xác định các vùng code "hotspot"
- Dự đoán mức độ khẩn cấp (Urgency)
- Dự đoán mức độ hoàn thiện (Completeness)
- Ước tính công sức cần thiết (Effort)

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

### Dự đoán cho một commit

```bash
python main.py predict --model_path path/to/model_checkpoint.pt --commit_message "fix: resolve authentication bug in login module"
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

Mô hình fusion đa phương thức kết hợp:

- **Text Encoder**: LSTM hai chiều hoặc Transformer để mã hóa commit messages
- **Metadata Encoder**: MLP để mã hóa metadata
- **Fusion Module**: Multi-head Attention để kết hợp các đặc trưng
- **Task Heads**: Các đầu dự đoán riêng cho từng nhiệm vụ (classification hoặc regression)

### Dữ liệu đầu vào

- **Text**: Commit messages
- **Metadata**: Thông tin về commit như tác giả, số lượng file thay đổi, số dòng thêm/xóa, thời gian commit, branch, v.v.

### Đầu ra

- **Risk**: Mức độ rủi ro (Low/Medium/High)
- **Complexity**: Độ phức tạp (Simple/Moderate/Complex)
- **Hotspot**: Mức độ hotspot (Low/Medium/High)
- **Urgency**: Mức độ khẩn cấp (Low/Medium/High)
- **Completeness**: Mức độ hoàn thiện (Partial/Complete/Final)
- **Estimated Effort**: Ước tính công sức (giá trị liên tục)

## Đóng góp

Các đóng góp, báo cáo lỗi và đề xuất cải tiến luôn được chào đón!

## Giấy phép

[MIT License](LICENSE)
