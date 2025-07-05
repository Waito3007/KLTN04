backendMultifusion
==================

Hướng dẫn triển khai và sử dụng các tính năng AI của mô hình MultiFusion
-----------------------------------------------------------------------

Tài liệu này mô tả cách triển khai, sử dụng và tích hợp mô hình MultiFusion cho phân loại commit/code đa ngôn ngữ, đa đặc trưng trong backend Python.

1. Tổng quan mô hình
--------------------
- MultiFusion là mô hình kết hợp BERT (DistilBERT) với các đặc trưng số (lines_added, files_count, ...), đặc trưng ngôn ngữ (one-hot), và commit message.
- Ứng dụng: Phân loại commit tự động (feat, fix, refactor, chore, ...), hỗ trợ kiểm duyệt, thống kê, gợi ý nhãn tự động cho hệ thống quản lý mã nguồn.

2. Chuẩn bị môi trường
----------------------
- Python >= 3.8
- torch, transformers, scikit-learn, numpy
- Đảm bảo có GPU (nếu muốn inference nhanh)

Cài đặt:
::

    pip install torch transformers scikit-learn numpy

3. Input/Output của mô hình
---------------------------

**Input:**
    - commit_message: str
    - lines_added: int
    - lines_removed: int
    - files_count: int
    - (optionally) metadata: {detected_language: str}

**Output:**
    - predicted_class: str (feat, fix, refactor, ...)
    - confidence: float (0-1)

**Ví dụ input:**
::

    {
        "commit_message": "Move Actuator Jersey infrastructure to spring-boot-jersey",
        "lines_added": 363,
        "lines_removed": 187,
        "files_count": 34,
        "metadata": {"detected_language": "Java"}
    }

**Ví dụ output:**
::

    {
        "predicted_class": "refactor",
        "confidence": 0.75
    }

4. Hướng dẫn sử dụng mô hình đã train
-------------------------------------

**Bước 1: Tải model và các encoder**
::

    from transformers import DistilBertTokenizer
    import torch
    from train_multifusion_v2 import MultiFusionModel
    # ... load label_encoder_lang, label_encoder_type, scaler như khi train ...
    model = MultiFusionModel(...)
    model.load_state_dict(torch.load('multifusionV2.pth', weights_only=True))
    model.eval()

**Bước 2: Tiền xử lý input**
::

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoding = tokenizer.encode_plus(
        commit_message,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    # numerical_features = scaler.transform([[lines_added, lines_removed, files_count, total_changes, ratio]])
    # lang_one_hot = ...

**Bước 3: Dự đoán**
::

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, numerical_features, language_one_hot)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
        predicted_class = label_encoder_type.inverse_transform([predicted_class_idx])[0]

5. Tích hợp API backend
-----------------------
- Có thể đóng gói thành REST API (Flask, FastAPI) nhận input JSON, trả về nhãn dự đoán và độ tự tin.
- Đảm bảo load model, encoder, scaler 1 lần khi khởi động server để tối ưu hiệu năng.

6. Lưu ý khi triển khai thực tế
-------------------------------
- Đảm bảo input được chuẩn hóa giống như khi train (đặc biệt là scaler, encoder).
- Nếu có class mới hoặc dữ liệu mới, cần retrain hoặc update encoder.
- Có thể batch inference để tăng tốc nếu cần dự đoán nhiều commit cùng lúc.

7. Liên hệ & đóng góp
---------------------
- Tác giả: [Điền tên bạn]
- Đóng góp, phản hồi: [Điền email hoặc github]

8. Các tính năng backend AI đề xuất triển khai
---------------------------------------------

**1. API phân loại commit/code tự động**
- Nhận input JSON (commit message, số dòng, số file, ngôn ngữ, ...)
- Trả về nhãn phân loại (feat, fix, refactor, chore, ...), độ tự tin
- Hỗ trợ batch inference (dự đoán nhiều commit cùng lúc)

**2. API gợi ý nhãn khi nhập commit message**
- Khi user nhập commit message trên giao diện, backend gọi AI để gợi ý nhãn phù hợp
- Trả về top-1 hoặc top-N nhãn kèm xác suất

**3. API kiểm duyệt commit/code**
- Phát hiện commit bất thường (ví dụ: nhãn other_type, nhãn không rõ ràng, nghi ngờ spam)
- Đề xuất review thủ công hoặc tự động flag

**4. API thống kê, phân tích lịch sử commit/code**
- Thống kê tỷ lệ các loại commit theo thời gian, theo user, theo repo
- Phân tích xu hướng refactor, bugfix, feature, ...
- Trả về dữ liệu dạng bảng hoặc biểu đồ cho dashboard frontend

**5. API kiểm tra chất lượng nhãn (label quality audit)**
- So sánh nhãn AI dự đoán với nhãn gán tay
- Phát hiện các commit có khả năng bị gán nhãn sai để review lại

**6. API hỗ trợ gán nhãn tự động cho dữ liệu mới**
- Khi crawl dữ liệu mới, backend tự động gán nhãn bằng AI
- Lưu nhãn AI vào DB, cho phép chỉnh sửa thủ công nếu cần

**7. API explainable AI (giải thích dự đoán)**
- Trả về lý do dự đoán (ví dụ: từ khóa trong message, số file lớn, ...)
- Hỗ trợ debug, tăng độ tin cậy khi tích hợp vào workflow thực tế

**8. API quản lý model**
- Kiểm tra version model đang deploy
- Reload model khi có model mới
- Log các request inference để phục vụ retrain hoặc audit

**9. API kiểm tra hiệu năng model (healthcheck, benchmark)**
- Đo thời gian inference, memory usage
- Trả về trạng thái sẵn sàng của AI backend

**10. API hỗ trợ retrain/finetune**
- Nhận dữ liệu mới, cho phép trigger retrain hoặc finetune model
- Lưu lại các phiên bản model, cho phép rollback nếu cần

**Lưu ý triển khai:**
- Tất cả API nên chuẩn RESTful, trả về JSON, log đầy đủ request/response
- Đảm bảo bảo mật, phân quyền khi truy cập các API nhạy cảm (retrain, quản lý model)
- Có thể tích hợp với hệ thống CI/CD để tự động deploy model mới

**Ví dụ endpoint:**
::

    POST /api/ai/classify_commit
    POST /api/ai/batch_classify
    POST /api/ai/suggest_label
    POST /api/ai/audit_label
    GET  /api/ai/stats
    POST /api/ai/explain
    POST /api/ai/retrain
    GET  /api/ai/model_version
    GET  /api/ai/health

