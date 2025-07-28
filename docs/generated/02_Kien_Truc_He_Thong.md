# Kiến trúc hệ thống

_Tài liệu này mô tả chi tiết về kiến trúc của dự án, bao gồm sự tương tác giữa frontend, backend và các dịch vụ AI._

## 1. Kiến trúc tổng thể

Hệ thống được thiết kế với kiến trúc hiện đại, tách biệt rõ ràng, bao gồm ba thành phần chính:

1.  **Frontend**: Một ứng dụng trang đơn (SPA) dựa trên React, đóng vai trò là giao diện người dùng.
2.  **Backend**: Một ứng dụng FastAPI cung cấp RESTful API, quản lý logic nghiệp vụ và lưu trữ dữ liệu.
3.  **AI Services**: Một tập hợp các dịch vụ Python đóng gói các mô hình học máy để phân tích thông minh.

![Sơ đồ kiến trúc tổng thể](https://i.imgur.com/9yZ4q4A.png)

## 2. Kiến trúc Backend

Backend được xây dựng bằng FastAPI và tuân theo mô hình thiết kế hướng dịch vụ (service-oriented). Điều này thúc đẩy tính module hóa, phân tách các mối quan tâm (separation of concerns) và dễ dàng bảo trì.

### 2.1. Các thành phần cốt lõi

- **`main.py`**: Điểm khởi đầu của ứng dụng. Nó khởi tạo ứng dụng FastAPI, thiết lập middleware (CORS) và bao gồm tất cả các API router.
- **`api/routes/`**: Thư mục này chứa các API endpoint, được nhóm theo chức năng. Mỗi file (ví dụ: `commit_routes.py`, `auth.py`) định nghĩa một tập hợp các route liên quan. Các route này chịu trách nhiệm xử lý các yêu cầu HTTP đến, xác thực dữ liệu bằng Pydantic model và gọi các service thích hợp.
- **`services/`**: Đây là nơi chứa logic nghiệp vụ cốt lõi. Các service chịu trách nhiệm điều phối các tác vụ, tương tác với các API bên ngoài (như GitHub) và giao tiếp với cơ sở dữ liệu. Các service chính bao gồm:
    - **`github_service.py`**: Một lớp "wrapper" bao quanh GitHub API. Nó lấy dữ liệu thô và làm giàu nó với các siêu dữ liệu bổ sung trước khi chuyển cho các service khác.
    - **`commit_service.py`**: Quản lý việc lưu trữ và truy xuất dữ liệu commit từ cơ sở dữ liệu. Nó bao gồm logic để xử lý hàng loạt (batch processing) và phân tích dữ liệu.
    - **`han_ai_service.py`**: Cung cấp một giao diện cho mô hình HAN để phân loại commit. Nó được thiết kế theo mẫu singleton để đảm bảo mô hình chỉ được tải một lần.
- **`db/`**: Thư mục này xử lý tất cả các hoạt động liên quan đến cơ sở dữ liệu.
    - **`database.py`**: Cấu hình kết nối cơ sở dữ liệu (PostgreSQL) bằng SQLAlchemy.
    - **`models/`**: Định nghĩa schema cơ sở dữ liệu bằng SQLAlchemy.
    - **`migrations/`**: Chứa các script migration của Alembic để quản lý các thay đổi schema.
- **`core/`**: Chứa các cài đặt ứng dụng cốt lõi, chẳng hạn như quản lý cấu hình (`config.py`), cài đặt bảo mật (`security.py`).
- **`schemas/`**: Chứa các Pydantic model được sử dụng để xác thực và tuần tự hóa dữ liệu trong lớp API.

### 2.2. Luồng dữ liệu trong Backend

Một yêu cầu đồng bộ hóa kho chứa điển hình sẽ tuân theo luồng sau:

1.  Một yêu cầu HTTP được gửi đến một endpoint cụ thể trong `api/routes/` (ví dụ: `/api/sync/repository/{repo_id}`).
2.  Route handler gọi `github_service` để lấy các commit mới nhất từ GitHub API.
3.  `github_service` lấy dữ liệu, xử lý và làm giàu nó với các chi tiết bổ sung (ví dụ: thống kê file).
4.  Dữ liệu đã được làm giàu sau đó được chuyển đến `commit_service`.
5.  `commit_service` lưu các commit mới vào cơ sở dữ liệu PostgreSQL, tránh trùng lặp.
6.  Đối với mỗi commit mới, `han_ai_service` được gọi để phân tích commit message.
7.  Kết quả phân tích được lưu trữ trong cơ sở dữ liệu, được liên kết với commit tương ứng.
8.  API endpoint trả về một phản hồi thành công cho client.

## 3. Kiến trúc Frontend

Frontend là một ứng dụng trang đơn hiện đại được xây dựng bằng React và Vite.

- **`src/`**: Thư mục mã nguồn chính.
  - **`pages/`**: Chứa các component cấp cao nhất cho mỗi trang (ví dụ: `Dashboard.jsx`, `LoginPage.jsx`).
  - **`components/`**: Chứa các component UI có thể tái sử dụng (ví dụ: `Button.jsx`, `Chart.jsx`).
  - **`api/`**: Chứa các hàm để thực hiện các cuộc gọi API đến backend, thường sử dụng thư viện như `axios`.
  - **`contexts/`**: Quản lý trạng thái toàn cục bằng Context API của React (ví dụ: để xác thực người dùng).
  - **`routes/`**: Định nghĩa định tuyến của ứng dụng bằng `react-router-dom`.
  - **`hooks/`**: Chứa các custom React hook để quản lý logic của component.

## 4. Kiến trúc Dịch vụ AI

Các dịch vụ AI được tích hợp trực tiếp vào backend nhưng được thiết kế theo dạng module.

- **`han_ai_service.py`**: Dịch vụ này đóng gói mô hình Hierarchical Attention Network (HAN).
    - **Tải mô hình**: Dịch vụ tải mô hình PyTorch và tokenizer đã được huấn luyện trước từ thư mục `backend/ai/models/`.
    - **Mẫu Singleton**: Nó sử dụng mẫu singleton để đảm bảo rằng mô hình chỉ được tải vào bộ nhớ một lần, điều này rất quan trọng đối với hiệu suất và quản lý tài nguyên.
    - **API dự đoán**: Nó cung cấp một phương thức đơn giản (`analyze_commit_message`) nhận một chuỗi và trả về một phân tích có cấu trúc, che giấu sự phức tạp của việc token hóa, chuyển đổi tensor và suy luận mô hình.
    - **Cơ chế Mock**: Nó bao gồm một cơ chế dự phòng để cung cấp phân tích giả lập nếu mô hình không tải được, giúp hệ thống trở nên linh hoạt hơn.

Thiết kế module này cho phép các mô hình AI mới có thể được tích hợp trong tương lai bằng cách chỉ cần tạo một lớp dịch vụ mới cho chúng.
