# Hướng dẫn sử dụng Script Gắn Nhãn Tự Động

Script `auto_labeling.py` được thiết kế để tự động gắn nhãn cho các commit trong dataset, nhằm hỗ trợ mô hình multimodal fusion gợi ý công việc cho các thành viên trong nhóm.

## Cài đặt Môi Trường

Trước khi sử dụng script, hãy cài đặt các thư viện cần thiết:

```bash
pip install nltk numpy
```

Các thư viện tùy chọn (để phân tích tốt hơn):

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Cách sử dụng

### Cách 1: Chỉ định file JSON đầu vào

```bash
python auto_labeling.py --input <đường_dẫn_đến_file_json>
```

Ví dụ:

```bash
python auto_labeling.py --input "data/processed/merged_commits_20250626_204114.json"
```

### Cách 2: Chỉ định cả file đầu vào và đầu ra

```bash
python auto_labeling.py --input <đường_dẫn_đến_file_json> --output <đường_dẫn_đến_file_đầu_ra>
```

Ví dụ:

```bash
python auto_labeling.py --input "data/processed/merged_commits_20250626_204114.json" --output "data/labeled/my_labeled_data.json"
```

### Cách 3: Chỉ định thư mục đầu ra

```bash
python auto_labeling.py --input <đường_dẫn_đến_file_json> --output-dir <thư_mục_đầu_ra>
```

Ví dụ:

```bash
python auto_labeling.py --input "data/processed/merged_commits_20250626_204114.json" --output-dir "data/custom_labeled"
```

### Chế độ mặc định (không tham số)

Nếu chạy không có tham số nào, script sẽ tự động tìm tất cả các file JSON trong thư mục `data/processed` có chứa "merged_commits" trong tên:

```bash
python auto_labeling.py
```

## Các nhãn được gắn

Script sẽ gắn các nhãn sau cho mỗi commit:

1. **Loại công việc** (task_type):

   - development, bug_fix, refactoring, documentation, testing, devops, security

2. **Độ phức tạp** (complexity):

   - low, medium, high

3. **Lĩnh vực kỹ thuật** (technical_area):

   - frontend, backend, database, infrastructure, mobile

4. **Kỹ năng yêu cầu** (required_skills):

   - python, javascript, react, angular, vue, java, dotnet, database, devops, testing

5. **Ưu tiên** (priority):
   - low, medium, high, critical

## Kết quả

Script sẽ tạo hai file:

1. **File JSON đã gắn nhãn**: Chứa toàn bộ dataset với các commit đã được gắn nhãn
2. **File thống kê** (`label_stats.json`): Chứa thống kê về phân bố các nhãn trong dataset

## Sử dụng cho mô hình multimodal fusion

Dataset đã gắn nhãn có thể được sử dụng để huấn luyện mô hình multimodal fusion nhằm:

1. Phân tích commit mới và gợi ý thành viên phù hợp nhất để xử lý
2. Ước tính độ phức tạp và ưu tiên của các công việc
3. Phân tích kỹ năng cần thiết cho mỗi loại công việc
4. Tự động phân loại và gắn nhãn cho các issue hoặc PR mới

## Xử lý lỗi

Nếu gặp lỗi khi chạy script, hãy kiểm tra:

1. Phiên bản Python (khuyến nghị: Python 3.8+)
2. Cấu trúc của file JSON đầu vào (phải có trường "data" chứa danh sách commit)
3. File nhật ký `auto_labeling.log` để biết thêm chi tiết

Nếu gặp lỗi với spaCy, bạn có thể chạy mà không cần cài đặt - script sẽ tự động chuyển sang chế độ NLTK.
