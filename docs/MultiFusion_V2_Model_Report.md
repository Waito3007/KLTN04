# Báo cáo Mô hình MultiFusion V2

## 1. Giới thiệu
Mô hình MultiFusion V2 là một hệ thống phân tích commit nâng cao, kết hợp sức mạnh của mô hình ngôn ngữ lớn (BERT) với các đặc trưng cấu trúc (numerical và categorical features) để phân loại loại commit, phân tích thành viên và cung cấp thông tin chi tiết về năng suất. Mô hình này được thiết kế để hiểu sâu sắc hơn về commit bằng cách xem xét cả nội dung văn bản và các số liệu liên quan đến mã nguồn.

## 2. Kiến trúc Mô hình (MultiFusionV2Model)
Mô hình `MultiFusionV2Model` được triển khai trong `backend/services/multifusion_v2_service.py` với kiến trúc chính như sau:

*   **Nhánh xử lý văn bản (Text Branch)**:
    *   Sử dụng `DistilBertModel` (một phiên bản nhỏ hơn của BERT) để trích xuất các đặc trưng ngữ nghĩa từ tin nhắn commit. BERT rất mạnh trong việc hiểu ngữ cảnh và ý nghĩa của văn bản.
    *   Đầu ra từ BERT được tổng hợp (pooled) để tạo ra một biểu diễn vector duy nhất cho tin nhắn commit.

*   **Nhánh xử lý đặc trưng cấu trúc (Structured Features Branch)**:
    *   Bao gồm một mạng nơ-ron đa lớp (MLP - Multi-Layer Perceptron) để xử lý các đặc trưng số (numerical features) và đặc trưng ngôn ngữ (language features).
    *   **Đặc trưng số**: Bao gồm số dòng code được thêm vào (`lines_added`), số dòng code bị xóa (`lines_removed`), số lượng tệp thay đổi (`files_count`), tổng số thay đổi (`total_changes`), và tỷ lệ thêm/xóa (`ratio_added_removed`). Các đặc trưng này được chuẩn hóa bằng `StandardScaler`.
    *   **Đặc trưng ngôn ngữ**: Ngôn ngữ lập trình được phát hiện trong commit (ví dụ: Python, JavaScript) được chuyển đổi thành biểu diễn one-hot encoding thông qua `LabelEncoder`.

*   **Lớp hợp nhất (Fusion Layer)**:
    *   Biểu diễn vector từ nhánh BERT và đầu ra từ nhánh MLP được nối (concatenate) lại với nhau.
    *   Một lớp phân loại tuyến tính (`nn.Linear`) được áp dụng trên các đặc trưng đã hợp nhất để dự đoán loại commit.

## 3. Đào tạo Mô hình
Thông tin chi tiết về quá trình đào tạo không được cung cấp trực tiếp trong `multifusion_v2_service.py`, nhưng có thể suy ra rằng mô hình được đào tạo trên một tập dữ liệu lớn các commit bao gồm cả tin nhắn commit và các số liệu liên quan đến mã nguồn. Quá trình này bao gồm:
*   **Tiền xử lý**: Token hóa tin nhắn commit bằng `DistilBertTokenizer`, chuẩn hóa các đặc trưng số và mã hóa one-hot các đặc trưng ngôn ngữ.
*   **Huấn luyện đa phương thức**: Mô hình học cách kết hợp thông tin từ cả văn bản và số liệu để đưa ra dự đoán chính xác.
*   **Lưu trữ**: Mô hình đã huấn luyện (`multifusionV2.pth`) và siêu dữ liệu (metadata_v2.json) chứa thông tin về các encoder và scaler được lưu trữ để sử dụng sau này.

## 4. Tích hợp và Sử dụng (MultiFusionV2Service)
Mô hình MultiFusion V2 được tích hợp vào hệ thống thông qua lớp `MultiFusionV2Service` (`backend/services/multifusion_v2_service.py`).
*   **Tải mô hình**: Dịch vụ tự động tải mô hình đã huấn luyện và siêu dữ liệu khi khởi tạo.
*   **Dự đoán loại commit (`predict_commit_type`)**: Đây là chức năng cốt lõi, nhận vào tin nhắn commit, số liệu thay đổi code và ngôn ngữ, sau đó trả về loại commit dự đoán cùng với độ tin cậy và phân phối xác suất cho tất cả các loại.
*   **Phân tích commit của thành viên (`analyze_member_commits`)**: Dịch vụ này có thể phân tích một tập hợp các commit của một thành viên, cung cấp các số liệu thống kê như phân phối loại commit, tổng số thay đổi, số lượng tệp đã sửa đổi, và ngôn ngữ được sử dụng.
*   **Phân tích hàng loạt (`batch_analyze_commits`)**: Cho phép phân tích nhiều tin nhắn commit cùng lúc.
*   **Thông tin mô hình (`get_model_info`)**: Cung cấp thông tin chi tiết về mô hình đã tải, bao gồm tên, kiến trúc, phiên bản, thiết bị đang chạy, các ngôn ngữ và loại commit được hỗ trợ, và các tính năng chính.

## 5. Mục đích và Ứng dụng
Mô hình MultiFusion V2 mang lại khả năng phân tích commit toàn diện hơn, hỗ trợ các tác vụ:
*   **Phân loại commit chính xác**: Kết hợp nhiều loại dữ liệu giúp cải thiện độ chính xác trong việc phân loại commit.
*   **Phân tích năng suất**: Cung cấp cái nhìn sâu sắc về năng suất của nhà phát triển thông qua các số liệu thay đổi code và phân phối loại commit.
*   **Hiểu hành vi phát triển**: Giúp xác định các mẫu hành vi của nhà phát triển và xu hướng phát triển của dự án.
*   **Hỗ trợ quản lý dự án**: Cung cấp dữ liệu phong phú hơn để đưa ra quyết định quản lý dự án tốt hơn.
