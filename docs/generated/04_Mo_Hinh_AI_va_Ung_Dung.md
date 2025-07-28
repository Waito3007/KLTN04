# Các mô hình AI và Ứng dụng

_Tài liệu này giải thích chi tiết về các mô hình Trí tuệ nhân tạo (AI) được sử dụng trong dự án, kiến trúc của chúng và cách chúng được tích hợp vào hệ thống._

## 1. Tổng quan về AI trong dự án

Trọng tâm của dự án là sử dụng AI để tự động hóa việc phân tích và hiểu dữ liệu từ các kho chứa Git. Thay vì chỉ dựa vào các thống kê cơ bản, hệ thống sử dụng các mô hình học máy để "đọc" và "hiểu" ý nghĩa đằng sau mỗi commit, từ đó cung cấp các thông tin chi tiết có giá trị.

Có hai mô hình AI chính trong dự án:

1.  **Hierarchical Attention Network (HAN)**: Mô hình chính đang được sử dụng để phân loại commit message.
2.  **MultiFusion V2**: Một mô hình thế hệ tiếp theo, đang trong giai đoạn phát triển, nhằm mục đích phân tích đa phương thức (multi-modal) để đưa ra các đánh giá sâu sắc hơn.

## 2. Mô hình HAN (Hierarchical Attention Network)

### 2.1. Mục tiêu

Mô hình HAN được sử dụng để giải quyết bài toán **phân loại văn bản** cho các commit message. Nó tự động gán một hoặc nhiều nhãn cho mỗi commit, chẳng hạn như:

-   **Loại thay đổi**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.
-   **Lĩnh vực ảnh hưởng**: `frontend`, `backend`, `database`, `testing`.
-   **Mức độ ảnh hưởng**: `low`, `medium`, `high`.

### 2.2. Kiến trúc

HAN là một mô hình mạng nơ-ron được thiết kế đặc biệt để nắm bắt cấu trúc phân cấp của văn bản. Một commit message (tài liệu) được xem như một tập hợp các câu, và mỗi câu là một tập hợp các từ.

Kiến trúc của nó bao gồm:

1.  **Lớp Biểu diễn Từ (Word Embedding)**: Chuyển mỗi từ thành một vector số, nắm bắt ngữ nghĩa của từ đó.
2.  **Bộ mã hóa Từ (Word Encoder)**: Sử dụng một mạng **GRU (Gated Recurrent Unit)** hai chiều để đọc các từ trong một câu và tạo ra một biểu diễn cho mỗi từ dựa trên ngữ cảnh của câu đó.
3.  **Cơ chế tập trung cấp Từ (Word-level Attention)**: Đây là phần quan trọng. Thay vì coi tất cả các từ trong câu là quan trọng như nhau, cơ chế này "học" cách gán trọng số cao hơn cho những từ quan trọng nhất để xác định ý nghĩa của câu. Ví dụ, trong câu "Fix critical bug in payment gateway", các từ "Fix", "critical", "bug", "payment" sẽ được chú ý nhiều hơn.
4.  **Bộ mã hóa Câu (Sentence Encoder)**: Sau khi mỗi câu được biểu diễn bằng một vector, một mạng GRU hai chiều khác được sử dụng để đọc các vector câu này và tạo ra biểu diễn cho toàn bộ commit message.
5.  **Cơ chế tập trung cấp Câu (Sentence-level Attention)**: Tương tự như cấp từ, cơ chế này học cách gán trọng số cao hơn cho những câu quan trọng nhất trong commit message để xác định ý nghĩa tổng thể của nó.
6.  **Lớp phân loại (Classification Layer)**: Cuối cùng, vector biểu diễn cho toàn bộ commit message được đưa qua một lớp mạng nơ-ron để phân loại ra các nhãn đã định nghĩa.

![Sơ đồ mô hình HAN](https://i.imgur.com/sY8gL9d.png)

### 2.3. Tích hợp vào hệ thống

-   **Vị trí**: Logic của mô hình được đóng gói trong `backend/services/han_ai_service.py`.
-   **Mô hình đã huấn luyện**: File mô hình được lưu tại `backend/ai/models/han_github_model/best_model.pth`.
-   **Quy trình**: Khi một commit mới được đồng bộ hóa, `commit_service` sẽ gọi `han_ai_service` để phân tích message của commit đó. Kết quả phân tích sau đó được lưu vào cơ sở dữ liệu.

## 3. Mô hình MultiFusion V2 (Đang phát triển)

### 3.1. Động lực

Mô hình HAN rất hiệu quả trong việc phân tích văn bản, nhưng nó có những hạn chế:

-   Nó chỉ xem xét **nội dung của commit message**.
-   Nó không thể đánh giá được **mức độ phức tạp hoặc rủi ro** của chính những thay đổi trong mã nguồn.
-   Một commit message được viết tốt có thể che giấu một thay đổi mã nguồn đầy rủi ro.

MultiFusion V2 được thiết kế để giải quyết những vấn đề này bằng cách tạo ra một mô hình **đa phương thức (multi-modal)**.

### 3.2. Kiến trúc dự kiến

Mô hình sẽ kết hợp (hợp nhất - fuse) dữ liệu từ ba luồng chính:

1.  **Dữ liệu văn bản**: 
    -   **Nguồn**: Commit message.
    -   **Bộ mã hóa**: Có thể là một mô hình ngôn ngữ lớn như BERT hoặc một phiên bản cải tiến của HAN.
2.  **Dữ liệu cấu trúc mã nguồn**:
    -   **Nguồn**: Các thay đổi trong code (diff), bao gồm các file đã thay đổi, số dòng thêm/xóa.
    -   **Bộ mã hóa**: Một mô hình chuyên dụng có thể xử lý cấu trúc code, ví dụ như Mạng nơ-ron đồ thị (GNN) trên Cây cú pháp trừu tượng (AST) của code, hoặc một mô hình đơn giản hơn sử dụng các đặc trưng như loại file, kích thước thay đổi, v.v.
3.  **Siêu dữ liệu về tác giả (Author Metadata)**:
    -   **Nguồn**: Thông tin về người thực hiện commit (ví dụ: tần suất commit, lịch sử gây lỗi).
    -   **Bộ mã hóa**: Một mạng nơ-ron truyền thẳng đơn giản để tạo ra một vector biểu diễn cho tác giả.

### 3.3. Lớp hợp nhất (Fusion Layer)

Các đầu ra từ ba bộ mã hóa trên sẽ được kết hợp lại bằng một cơ chế tập trung (attention). Lớp này sẽ học cách cân nhắc tầm quan trọng của mỗi phương thức (văn bản, code, tác giả) để đưa ra dự đoán cuối cùng.

### 3.4. Mục tiêu đầu ra

Mô hình sẽ được huấn luyện để dự đoán:

-   **Phân loại Commit**: Một phiên bản cải tiến của phân loại HAN.
-   **Điểm rủi ro (Risk Score)**: Một điểm số (ví dụ: từ 0-10) cho biết nguy cơ tiềm tàng của commit trong việc gây ra lỗi.
-   **Khu vực ảnh hưởng (Impact Area)**: Phần của ứng dụng bị ảnh hưởng nhiều nhất bởi thay đổi (ví dụ: 'UI', 'Database', 'API').

### 3.5. Tình trạng phát triển

-   **Vị trí**: Các thành phần liên quan được đặt trong `backend/ai/multimodal_fusion_network/`.
-   **Tình trạng**: Đang trong giai đoạn thu thập dữ liệu và thử nghiệm các bộ mã hóa riêng lẻ. `multifusion_ai_service.py` là một placeholder cho dịch vụ sẽ được phát triển trong tương lai.
