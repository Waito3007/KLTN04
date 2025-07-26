# Báo cáo Mô hình HAN (Hierarchical Attention Network)

## 1. Giới thiệu
Mô hình HAN (Hierarchical Attention Network) được sử dụng để phân tích và phân loại các tin nhắn commit trong dự án. Mô hình này đặc biệt hiệu quả trong việc xử lý văn bản có cấu trúc phân cấp (như tin nhắn commit có thể có nhiều dòng, mỗi dòng là một "câu"). HAN có khả năng tập trung vào các phần quan trọng của tin nhắn ở cả cấp độ từ (word-level) và cấp độ câu (sentence-level) để đưa ra dự đoán chính xác.

## 2. Kiến trúc Mô hình (SimpleHANModel)
Mô hình `SimpleHANModel` được triển khai trong `backend/ai/train_han_github.py` và `backend/services/han_ai_service.py` với kiến trúc chính như sau:

*   **Lớp Embedding**: Chuyển đổi các từ thành các vector dày đặc (dense vectors).
*   **LSTM cấp độ từ (Word-level LSTM)**: Xử lý các từ trong mỗi "câu" (dòng) của tin nhắn commit. Nó học biểu diễn ngữ cảnh của từng từ.
*   **Cơ chế Attention cấp độ từ (Word-level Attention)**: Gán trọng số cho các từ khác nhau trong một "câu", cho phép mô hình tập trung vào các từ quan trọng nhất để tạo ra biểu diễn "câu" (sentence vector).
*   **LSTM cấp độ câu (Sentence-level LSTM)**: Xử lý các "câu" (sentence vectors) trong toàn bộ tin nhắn commit. Nó học biểu diễn ngữ cảnh của từng "câu".
*   **Cơ chế Attention cấp độ câu (Sentence-level Attention)**: Gán trọng số cho các "câu" khác nhau trong tin nhắn commit, cho phép mô hình tập trung vào các "câu" quan trọng nhất để tạo ra biểu diễn tổng thể của tin nhắn (document vector).
*   **Các đầu phân loại đa nhiệm (Multi-task Classification Heads)**: Từ biểu diễn tổng thể của tin nhắn, mô hình sử dụng các lớp tuyến tính riêng biệt để thực hiện nhiều nhiệm vụ phân loại đồng thời, ví dụ:
    *   `commit_type`: Loại commit (feat, fix, docs, style, refactor, test, chore, other).
    *   `purpose`: Mục đích của commit.
    *   `sentiment`: Sắc thái của commit.
    *   `tech_tag`: Thẻ công nghệ liên quan.

## 3. Đào tạo Mô hình (train_han_github.py)
Quá trình đào tạo mô hình HAN được thực hiện bởi script `train_han_github.py`.
*   **Dữ liệu**: Sử dụng tập dữ liệu GitHub commits (từ `github_commits_training_data.json`).
*   **Tokenizer**: `SimpleTokenizer` được sử dụng để chuyển đổi văn bản thành các ID số, hỗ trợ xử lý phân cấp (tách câu và từ).
*   **Multi-task Learning**: Mô hình được đào tạo để dự đoán nhiều thuộc tính của commit cùng một lúc, giúp cải thiện hiệu suất tổng thể và khả năng khái quát hóa.
*   **Tối ưu hóa**: Sử dụng `AdamW` optimizer và `CrossEntropyLoss` cho từng nhiệm vụ. Hỗ trợ tối ưu hóa GPU với `torch.cuda.amp.GradScaler` để tăng tốc độ đào tạo.
*   **Lưu trữ**: Mô hình tốt nhất (dựa trên độ chính xác trên tập validation) được lưu lại dưới dạng `best_model.pth`.

## 4. Tích hợp và Sử dụng (HANAIService)
Mô hình HAN được tích hợp vào hệ thống thông qua lớp `HANAIService` (`backend/services/han_ai_service.py`).
*   **Tải mô hình**: Dịch vụ tự động tải mô hình HAN đã được đào tạo (`best_model.pth`) khi khởi tạo.
*   **Tiền xử lý**: Tin nhắn commit được tiền xử lý (token hóa, chuyển đổi thành tensor) trước khi đưa vào mô hình.
*   **Dự đoán**: Phương thức `_predict_with_han_model` thực hiện dự đoán trên tin nhắn commit và trả về các nhãn dự đoán cùng với độ tin cậy cho từng nhiệm vụ (loại commit, khu vực công nghệ, tác động).
*   **Phân tích**: `analyze_commit_message` là hàm chính để phân tích một tin nhắn commit, cung cấp kết quả phân loại chi tiết.
*   **Ứng dụng**: Dịch vụ này được sử dụng để:
    *   Phân tích từng commit.
    *   Phân tích hàng loạt commit (`analyze_commits_batch`).
    *   Phân tích mẫu commit của nhà phát triển (`analyze_developer_patterns`).
    *   Đề xuất phân công nhiệm vụ (`suggest_task_assignment`).
    *   Tạo thông tin chi tiết về dự án (`generate_project_insights`).

## 5. Mục đích và Ứng dụng
Mô hình HAN giúp tự động phân loại và hiểu sâu hơn về nội dung của các tin nhắn commit. Điều này hỗ trợ các tác vụ quản lý dự án như:
*   **Phân loại commit**: Tự động gán loại (feature, fix, docs, v.v.) cho commit.
*   **Đánh giá tác động**: Ước tính tác động của commit (thấp, trung bình, cao).
*   **Phân tích chuyên môn**: Xác định lĩnh vực chuyên môn của nhà phát triển dựa trên các commit của họ.
*   **Quản lý dự án**: Cung cấp thông tin chi tiết về sức khỏe dự án, xu hướng chất lượng mã và đề xuất cải tiến.
