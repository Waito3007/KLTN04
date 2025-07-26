# Tài liệu Mô hình AI trong Hệ thống

## Tổng quan

Hệ thống tích hợp hai mô hình AI chính để phân tích commit GitHub:

1. **HAN Model (Hierarchical Attention Network)** - Phân tích văn bản đa cấp
2. **MultiFusion V2 Model** - Kết hợp văn bản với các đặc trưng cấu trúc

## 1. Mô hình HAN (Hierarchical Attention Network)

### 1.1. Thu thập và Chuẩn bị Dữ liệu

#### Nguồn dữ liệu:

- **File**: `github_commits_training_data.json`
- **Nội dung**: Commit messages từ các repository GitHub
- **Cấu trúc dữ liệu**:

```json
{
  "message": "Fix bug in user authentication",
  "commit_type": "fix",
  "purpose": "bugfix",
  "sentiment": "neutral",
  "tech_tag": "backend"
}
```

#### Tiền xử lý dữ liệu:

- **Tokenization**: Sử dụng `SimpleTokenizer` để chuyển văn bản thành ID số
- **Phân cấp**: Tách commit message thành câu và từ
- **Encoding**: Chuyển đổi nhãn categorical thành số

### 1.2. Kiến trúc Mô hình

#### Cấu trúc HAN:

```python
class SimpleHANModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        # Word-level components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.word_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.word_attention = AttentionLayer(hidden_dim * 2)

        # Sentence-level components
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True)
        self.sentence_attention = AttentionLayer(hidden_dim * 2)

        # Multi-task heads
        self.commit_type_classifier = nn.Linear(hidden_dim * 2, num_commit_types)
        self.purpose_classifier = nn.Linear(hidden_dim * 2, num_purposes)
        self.sentiment_classifier = nn.Linear(hidden_dim * 2, num_sentiments)
        self.tech_tag_classifier = nn.Linear(hidden_dim * 2, num_tech_tags)
```

#### Đặc điểm chính:

- **Word-level Attention**: Tập trung vào từ quan trọng trong câu
- **Sentence-level Attention**: Tập trung vào câu quan trọng trong commit
- **Multi-task Learning**: Dự đoán đồng thời 4 thuộc tính

### 1.3. Quá trình Huấn luyện

#### Script huấn luyện: `train_han_github.py`

```python
# Cấu hình training
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

# Training loop với mixed precision
for epoch in range(num_epochs):
    for batch in train_loader:
        with torch.cuda.amp.autocast():
            outputs = model(batch_input)
            # Tính loss cho từng task
            loss_commit = criterion(outputs['commit_type'], targets['commit_type'])
            loss_purpose = criterion(outputs['purpose'], targets['purpose'])
            # ... các loss khác
            total_loss = loss_commit + loss_purpose + loss_sentiment + loss_tech

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
```

#### Hyperparameters:

- **Learning rate**: 0.001
- **Batch size**: 32
- **Hidden dimensions**: 128
- **Embedding dimensions**: 100
- **Optimizer**: AdamW với weight decay 0.01

### 1.4. Đánh giá và Kiểm thử

#### Metrics đánh giá:

- **Accuracy**: Độ chính xác cho từng task
- **F1-score**: Đánh giá cân bằng precision và recall
- **Confusion Matrix**: Phân tích chi tiết từng class

#### Kết quả mong đợi:

- **Commit Type Classification**: ~85-90% accuracy
- **Purpose Detection**: ~80-85% accuracy
- **Sentiment Analysis**: ~75-80% accuracy
- **Tech Tag Classification**: ~70-75% accuracy

#### Kiểm thử:

```python
# Test individual prediction
han_service = HANAIService()
result = han_service.analyze_commit_message("Fix authentication bug in login module")
# Expected output:
# {
#   "commit_type": "fix",
#   "confidence": 0.95,
#   "purpose": "bugfix",
#   "sentiment": "neutral",
#   "tech_tag": "backend"
# }
```

## 2. Mô hình MultiFusion V2

### 2.1. Thu thập và Chuẩn bị Dữ liệu

#### Nguồn dữ liệu:

- **Commit messages**: Văn bản mô tả commit
- **Code metrics**: Lines added, lines removed, files count
- **Language detection**: Ngôn ngữ lập trình chính
- **Commit types**: 11 loại commit (feat, fix, docs, style, refactor, test, chore, build, ci, perf, other)

#### Cấu trúc dữ liệu training:

```json
{
  "message": "Add user authentication endpoint",
  "lines_added": 120,
  "lines_removed": 15,
  "files_count": 3,
  "detected_language": "python",
  "commit_type": "feat"
}
```

#### Feature Engineering:

```python
# Numerical features (5 features)
numerical_features = [
    lines_added,
    lines_removed,
    files_count,
    total_changes,      # lines_added + lines_removed
    ratio_added_removed # lines_added / lines_removed
]

# Language features (one-hot encoding)
supported_languages = ['python', 'javascript', 'java', 'cpp', 'unknown_language']
language_one_hot = encode_language(detected_language)
```

### 2.2. Kiến trúc Mô hình

#### Cấu trúc MultiFusion V2:

```python
class MultiFusionV2Model(nn.Module):
    def __init__(self, bert_model_name, num_numerical_features, num_language_features, num_classes):
        # Text branch - BERT
        self.bert = DistilBertModel.from_pretrained(bert_model_name)

        # Structured features branch - MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_numerical_features + num_language_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion layer
        self.classifier = nn.Linear(self.bert.config.hidden_size + 64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, numerical_features, language_one_hot):
        # Text branch
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0, :]  # [CLS] token

        # Structured features branch
        combined_structured_features = torch.cat((numerical_features, language_one_hot), dim=1)
        mlp_output = self.mlp(combined_structured_features)

        # Fusion
        combined_features = torch.cat((pooled_output, mlp_output), dim=1)
        return self.classifier(self.dropout(combined_features))
```

#### Đặc điểm chính:

- **Multi-modal**: Kết hợp text (BERT) + structured features (MLP)
- **BERT-based**: Sử dụng DistilBERT cho hiểu ngữ cảnh tốt
- **Feature fusion**: Late fusion strategy

### 2.3. Quá trình Huấn luyện

#### Data preprocessing:

```python
# Text preprocessing
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encoding = tokenizer.encode_plus(
    commit_message,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Numerical features scaling
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)

# Language encoding
label_encoder_lang = LabelEncoder()
language_encoded = label_encoder_lang.fit_transform(languages)
```

#### Training configuration:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training với early stopping
best_accuracy = 0.0
patience = 5
patience_counter = 0
```

#### Metadata lưu trữ:

```json
{
  "num_numerical_features": 5,
  "num_language_features": 5,
  "num_classes": 11,
  "language_classes": [
    "python",
    "javascript",
    "java",
    "cpp",
    "unknown_language"
  ],
  "commit_type_classes": [
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "test",
    "chore",
    "build",
    "ci",
    "perf",
    "other"
  ],
  "scaler_mean": [10.0, 5.0, 2.0, 15.0, 2.0],
  "scaler_scale": [50.0, 25.0, 5.0, 75.0, 10.0]
}
```

### 2.4. Đánh giá và Kiểm thử

#### Metrics đánh giá:

- **Accuracy**: Độ chính xác tổng thể
- **Per-class F1**: F1-score cho từng loại commit
- **Confusion Matrix**: Ma trận nhầm lẫn chi tiết
- **Confidence Distribution**: Phân phối độ tin cậy

#### Kết quả mong đợi:

- **Overall Accuracy**: ~92-95%
- **Weighted F1-score**: ~90-93%
- **High-confidence predictions**: >80% predictions với confidence >0.8

#### Kiểm thử chi tiết:

```python
# Test single prediction
multifusion_service = MultiFusionV2Service()
result = multifusion_service.predict_commit_type(
    commit_message="Add user authentication with JWT tokens",
    lines_added=85,
    lines_removed=12,
    files_count=4,
    detected_language="python"
)

# Expected output:
# {
#   "commit_type": "feat",
#   "confidence": 0.94,
#   "all_probabilities": {
#     "feat": 0.94,
#     "fix": 0.03,
#     "docs": 0.01,
#     ...
#   },
#   "input_features": {
#     "lines_added": 85,
#     "lines_removed": 12,
#     "files_count": 4,
#     "detected_language": "python",
#     "total_changes": 97,
#     "ratio_added_removed": 7.08
#   }
# }

# Test batch analysis
commits_data = [...]  # List of commit data
analysis = multifusion_service.analyze_member_commits(commits_data)
# Returns detailed statistics and insights
```

## 3. So sánh và Lựa chọn Mô hình

### 3.1. So sánh Performance

| Aspect               | HAN Model                     | MultiFusion V2                 |
| -------------------- | ----------------------------- | ------------------------------ |
| **Input Types**      | Text only                     | Text + Code metrics + Language |
| **Architecture**     | Hierarchical LSTM + Attention | BERT + MLP Fusion              |
| **Tasks**            | Multi-task (4 outputs)        | Single-task (commit type)      |
| **Accuracy**         | ~85-90%                       | ~92-95%                        |
| **Inference Speed**  | Fast                          | Moderate                       |
| **Memory Usage**     | Low                           | High (due to BERT)             |
| **Interpretability** | High (attention weights)      | Moderate                       |

### 3.2. Use Cases

#### HAN Model phù hợp cho:

- Phân tích đa chiều commit (type, purpose, sentiment, tech)
- Hiểu attention patterns trong text
- Môi trường resource-constrained
- Cần giải thích chi tiết

#### MultiFusion V2 phù hợp cho:

- Phân loại commit type chính xác cao
- Có sẵn code metrics và language info
- Phân tích productivity và patterns
- Cần insights về coding behavior

## 4. Deployment và Monitoring

### 4.1. Model Serving

```python
# Service integration
from services.han_ai_service import HANAIService
from services.multifusion_v2_service import MultiFusionV2Service

# API endpoints
@app.post("/api/analyze/commit/han")
async def analyze_with_han(request: CommitAnalysisRequest):
    han_service = HANAIService()
    return han_service.analyze_commit_message(request.message)

@app.post("/api/analyze/commit/multifusion")
async def analyze_with_multifusion(request: DetailedCommitRequest):
    mf_service = MultiFusionV2Service()
    return mf_service.predict_commit_type(
        request.message,
        request.lines_added,
        request.lines_removed,
        request.files_count,
        request.detected_language
    )
```

### 4.2. Model Health Monitoring

- **Prediction confidence distribution**
- **Response time metrics**
- **Error rate tracking**
- **Model availability checks**
- **Resource usage monitoring**

### 4.3. Update và Retraining

- **Incremental learning** với new commit data
- **A/B testing** cho model updates
- **Performance degradation detection**
- **Automated retraining pipelines**

## 5. Kết luận

Hệ thống sử dụng hai mô hình bổ trợ cho nhau:

- **HAN** cho phân tích sâu văn bản và multi-task insights
- **MultiFusion V2** cho classification chính xác với multi-modal data

Cả hai mô hình đều được tích hợp chặt chẽ vào backend system và cung cấp APIs để frontend sử dụng, tạo nên một hệ thống phân tích commit toàn
