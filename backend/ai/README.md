# 🤖 AI Commit Analysis System

Hệ thống phân tích commit sử dụng Hierarchical Attention Network (HAN) để phân loại commit message và phân tích hoạt động của team.

## 📁 Cấu trúc thư mục

```
ai/
├── README.md                     # Hướng dẫn này
├── train_han_github.py           # Script train model HAN
├── test_commit_analyzer.py       # Script test và phân tích commit
├── simple_advanced_analysis.py   # Phân tích chi tiết và recommendations
├── simple_dataset_creator.py     # Tạo dataset để train
├── models/                       # Thư mục chứa model đã train
│   └── han_github_model/
│       └── best_model.pth        # Model HAN đã train xong
├── test_results/                 # Kết quả test và báo cáo
└── analysis_plots/              # Biểu đồ phân tích (nếu có)
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
cd backend/ai
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib seaborn scikit-learn numpy pandas
```

### 2. Train Model (Tùy chọn)

Model đã được train sẵn tại `models/han_github_model/best_model.pth`. Nếu muốn train lại:

```bash
python train_han_github.py
```

**Chi tiết quá trình train:**
- Model: Hierarchical Attention Network (HAN)
- Tasks: Multi-task learning cho 4 task:
  - `commit_type`: feat, fix, docs, style, refactor, test, chore, perf
  - `purpose`: Feature Implementation, Bug Fix, Documentation Update, etc.
  - `sentiment`: positive, negative, neutral, urgent
  - `tech_tag`: general, frontend, backend, database, etc.
- Accuracy đạt được: **98.95%**
- GPU Support: CUDA-enabled

### 3. Test và Phân tích Commit

**Chạy phân tích cơ bản:**
```bash
python test_commit_analyzer.py
```

**Kết quả sẽ bao gồm:**
- ✅ Phân tích 21 sample commits
- 📊 Thống kê loại commit (feat: 38.1%, fix: 28.6%, etc.)
- 👥 Phân tích hoạt động author (high/normal/low activity)
- 🎯 Confidence scores (trung bình 99.2%)
- 📁 Báo cáo JSON chi tiết

**Chạy phân tích nâng cao:**
```bash
python simple_advanced_analysis.py
```

**Kết quả nâng cao:**
- 🔍 Pattern analysis cho từng author
- 💡 Recommendations cụ thể
- ⚖️ Workload balance analysis
- 📋 Action plan với timeline

## 📊 Output Examples

### Basic Analysis Output:
```
📈 COMMIT ANALYSIS REPORT
=======================================
📊 OVERVIEW:
   Total commits analyzed: 21
   Unique authors: 5
   Average commits per author: 4.2

🏷️ COMMIT TYPE DISTRIBUTION:
   feat: 8 (38.1%)
   fix: 6 (28.6%)
   style: 2 (9.5%)

👥 AUTHOR ACTIVITY LEVELS:
   High: 1 authors (20.0%)
   Normal: 3 authors (60.0%)
   Low: 1 authors (20.0%)

🏆 TOP CONTRIBUTORS:
    1. John Doe: 10 commits (high)
    2. Alice Johnson: 4 commits (normal)
    3. Charlie Brown: 1 commits (low)
```

### Advanced Analysis Output:
```
💡 KHUYẾN NGHỊ CHI TIẾT CHO TEAM
================================
⚖️ PHÂN TÍCH WORKLOAD:
   📊 Tỷ lệ workload: 10.0:1
   ⚠️ CẢNH BÁO: Workload không cân bằng!

💤 HOẠT ĐỘNG THẤP (1 dev):
   💤 Charlie Brown: 1 commits (24% của trung bình)
   💡 Khuyến nghị:
      - Kiểm tra workload và obstacles
      - Cung cấp mentoring hoặc training

📋 ACTION PLAN:
   1. [MEDIUM] 1-on-1s with low-activity developers
   2. [HIGH] Review workload distribution
```

## 🛠️ Customization

### Thêm commit mới để test:

Chỉnh sửa trong `test_commit_analyzer.py`:

```python
sample_commits = [
    {
        "text": "feat: add new user authentication system",
        "author": "Your Name",
        "timestamp": "2025-06-07 10:00:00"
    },
    # Thêm commits khác...
]
```

### Điều chỉnh activity thresholds:

```python
# Trong hàm generate_author_stats()
if commit_count < avg_commits_per_author * 0.5:
    activity_level = "low"       # < 50% average
elif commit_count < avg_commits_per_author * 1.5:
    activity_level = "normal"    # 50-150% average  
elif commit_count < avg_commits_per_author * 3:
    activity_level = "high"      # 150-300% average
else:
    activity_level = "overloaded" # > 300% average
```

## 🎯 Use Cases

### 1. Team Performance Monitoring
- Phát hiện team members overloaded hoặc underperforming
- Track commit quality và patterns
- Monitor team morale qua sentiment analysis

### 2. Code Review Insights  
- Tự động phân loại commits theo type và purpose
- Identify commits cần review kỹ (low confidence)
- Track testing và documentation coverage

### 3. Project Management
- Workload balancing recommendations
- Identify bottlenecks trong development process
- Generate reports cho stakeholders

## 📈 Model Performance

- **Overall Accuracy**: 98.95%
- **Average Confidence**: 99.2%
- **Supported Commit Types**: 8 types (feat, fix, docs, etc.)
- **Multi-task Learning**: 4 simultaneous predictions
- **GPU Accelerated**: RTX 3050 support confirmed

## 🔧 Troubleshooting

### Model không load được:
```bash
# Kiểm tra file model tồn tại
ls models/han_github_model/best_model.pth

# Kiểm tra PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### CUDA issues:
```bash
# Install CUDA-compatible PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory issues:
- Giảm batch size trong train script
- Sử dụng CPU thay vì GPU: set `device = 'cpu'`

## 📝 Notes

- Model được train trên conventional commit format
- Sample data được tạo với patterns cụ thể để test
- Có thể integrate với Git hooks để real-time analysis
- Support cả English và Vietnamese output

## 🤝 Contributing

Để cải thiện model:
1. Thêm training data trong `simple_dataset_creator.py`
2. Điều chỉnh model architecture trong `train_han_github.py`
3. Test với `test_commit_analyzer.py`
4. Chạy advanced analysis để verify improvements

---

**Created by**: AI Team  
**Last Updated**: June 7, 2025  
**Model Version**: HAN v1.0  
**Accuracy**: 98.95%
