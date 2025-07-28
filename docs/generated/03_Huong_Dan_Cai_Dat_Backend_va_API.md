# Hướng dẫn Backend: Cài đặt và API

_Tài liệu này cung cấp hướng dẫn chi tiết để cài đặt môi trường backend và tham khảo các API endpoint chính._

## 1. Cài đặt Backend

Thực hiện theo các bước sau để chạy server backend trên máy cục bộ của bạn.

### Yêu cầu

- Python 3.12+
- Poetry (để quản lý các gói phụ thuộc)
- Một instance PostgreSQL đang chạy

### Các bước cài đặt

1.  **Di chuyển đến thư mục Backend**
    ```bash
    cd backend/
    ```

2.  **Cài đặt các gói phụ thuộc**
    Sử dụng Poetry để cài đặt tất cả các gói Python cần thiết.
    ```bash
    poetry install
    ```

3.  **Cấu hình biến môi trường**
    Tạo một file `.env` trong thư mục `backend/`. Bạn có thể sao chép từ file `.env.example` nếu có. File này lưu trữ các cấu hình nhạy cảm.

    **Các biến bắt buộc:**
    ```dotenv
    # Chuỗi kết nối đầy đủ đến cơ sở dữ liệu PostgreSQL của bạn
    DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>:<port>/<dbname>

    # Một khóa bí mật để ký JWT. Tạo một chuỗi ngẫu nhiên an toàn.
    SECRET_KEY=your_super_secret_key

    # GitHub Personal Access Token (PAT) của bạn có quyền truy cập kho chứa
    GITHUB_TOKEN=your_github_pat_here

    # Thông tin xác thực OAuth 2.0 để đăng nhập bằng GitHub
    GITHUB_CLIENT_ID=your_github_oauth_client_id
    GITHUB_CLIENT_SECRET=your_github_oauth_client_secret
    ```

4.  **Chạy Database Migrations**
    Alembic được sử dụng để quản lý schema của cơ sở dữ liệu. Chạy lệnh sau để áp dụng tất cả các migration và tạo các bảng cần thiết.
    ```bash
    alembic upgrade head
    ```

5.  **Khởi động Server**
    Sử dụng Uvicorn để chạy ứng dụng FastAPI. Cờ `--reload` cho phép tự động tải lại server khi có thay đổi trong mã nguồn để phát triển.
    ```bash
    uvicorn main:app --reload
    ```

    API sẽ có sẵn tại `http://127.0.0.1:8000`. Bạn có thể truy cập tài liệu API tương tác (Swagger UI) tại `http://127.0.0.1:8000/docs`.

## 2. Tham khảo các API Endpoint

Dưới đây là một số endpoint API cốt lõi do backend cung cấp.

*(Lưu ý: Tất cả các endpoint đều có tiền tố `/api`)*

### Xác thực (`auth.py`)

-   **`POST /auth/token`**: Đăng nhập người dùng. Yêu cầu `username` và `password` trong payload dạng form-data. Trả về một access token JWT.
-   **`GET /auth/github/login`**: Bắt đầu luồng đăng nhập GitHub OAuth2.
-   **`GET /auth/github/callback`**: URL callback để GitHub chuyển hướng đến sau khi ủy quyền. Xử lý việc tạo/đăng nhập người dùng và trả về token.

### Kho chứa (`repositories.py`)

-   **`POST /repositories/`**: Thêm một kho chứa mới vào hệ thống bằng cách cung cấp URL GitHub của nó.
-   **`GET /repositories/`**: Lấy danh sách tất cả các kho chứa do người dùng hiện tại thêm vào.
-   **`GET /repositories/{repo_id}`**: Lấy thông tin chi tiết cho một kho chứa cụ thể.

### Đồng bộ hóa (`sync.py`)

-   **`POST /sync/repository/{repo_id}`**: Kích hoạt quá trình đồng bộ hóa cho một kho chứa cụ thể. Thao tác này sẽ lấy các commit, branch mới nhất và các dữ liệu khác từ GitHub và lưu vào cơ sở dữ liệu.

### Commits (`commit_routes.py`)

-   **`GET /repositories/{repo_id}/commits`**: Lấy danh sách các commit (có phân trang) cho một kho chứa nhất định.
-   **`GET /commits/{commit_sha}`**: Lấy thông tin chi tiết cho một commit duy nhất bằng mã SHA của nó.
-   **`GET /repositories/{repo_id}/stats`**: Lấy các thống kê tổng hợp cho một kho chứa, chẳng hạn như tổng số commit, số dòng code thêm/xóa và số tác giả duy nhất.
-   **`GET /repositories/{repo_id}/trends`**: Cung cấp phân tích xu hướng commit trong một khoảng thời gian nhất định (ví dụ: 30 ngày qua).

### AI & Phân tích

-   **`GET /ai/status` (`ai_status.py`)**: Kiểm tra trạng thái của các mô hình AI (ví dụ: đã được tải thành công hay chưa).
-   **`GET /analysis/area/{repo_id}` (`area_analysis.py`)**: Thực hiện phân tích các lĩnh vực công việc khác nhau trong kho chứa dựa trên lịch sử commit (ví dụ: frontend, backend, docs).
-   **`GET /analysis/risk/{repo_id}` (`risk_analysis.py`)**: Cung cấp phân tích rủi ro cho kho chứa, có khả năng xác định các commit hoặc file có rủi ro cao (phụ thuộc vào mô hình MultiFusion V2).
-   **`POST /recommendations/assignment` (`assignment_recommendation.py`)**: Đề xuất các nhà phát triển phù hợp nhất cho một danh sách công việc dựa trên lịch sử đóng góp của họ.

Đây không phải là danh sách đầy đủ. Để có danh sách đầy đủ và tương tác của tất cả các endpoint có sẵn, vui lòng tham khảo Swagger UI tại `http://127.0.0.1:8000/docs` khi server đang chạy.
