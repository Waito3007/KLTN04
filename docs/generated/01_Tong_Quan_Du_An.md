# Tổng quan dự án

_Tài liệu này cung cấp cái nhìn tổng quan về dự án, mục tiêu, các tính năng chính và kiến trúc hệ thống._

## 1. Mục tiêu dự án

Mục tiêu chính của dự án là xây dựng một **trợ lý quản lý dự án thông minh**. Công cụ này sử dụng Trí tuệ nhân tạo (AI) để phân tích dữ liệu từ các kho chứa mã nguồn Git, qua đó cung cấp những hiểu biết sâu sắc về quy trình phát triển phần mềm, giúp các nhóm làm việc cải thiện hiệu suất, xác định rủi ro và tăng năng suất.

## 2. Các tính năng chính

- **Phân tích Commit tự động**: Hệ thống sử dụng mô hình **Hierarchical Attention Network (HAN)** để tự động phân loại các commit message theo các danh mục như `feat` (tính năng mới), `fix` (sửa lỗi), `docs` (tài liệu), `style` (định dạng code), `refactor` (tái cấu trúc), `test` (kiểm thử), và `chore` (công việc vặt). Điều này giúp cung cấp một cái nhìn rõ ràng về bản chất của các thay đổi trong dự án.
- **Dashboard tương tác**: Giao diện web hiển thị các số liệu và xu hướng quan trọng được phân tích từ dữ liệu kho chứa.
- **Làm giàu dữ liệu**: Hệ thống không chỉ lấy dữ liệu thô từ GitHub mà còn xử lý và tính toán thêm các chỉ số hữu ích như: các loại file đã thay đổi, các thư mục bị ảnh hưởng, và tổng số dòng code thay đổi trong mỗi commit.
- **Kiến trúc có khả năng mở rộng**: Backend được xây dựng với FastAPI và tuân theo kiến trúc hướng dịch vụ (service-oriented), giúp dễ dàng thêm các tính năng mới và mở rộng hệ thống trong tương lai.

## 3. Kiến trúc hệ thống

Dự án bao gồm ba thành phần chính:

- **Frontend**: Một ứng dụng trang đơn (SPA) được xây dựng bằng **React** và **Vite**. Đây là giao diện người dùng để tương tác với hệ thống.
- **Backend**: Một ứng dụng Python được xây dựng bằng **FastAPI**. Nó chịu trách nhiệm cung cấp API, xử lý logic nghiệp vụ và tích hợp với các mô hình AI.
- **AI Services**: Một tập hợp các dịch vụ chứa các mô hình học máy để thực hiện các tác vụ phân tích thông minh.

### 3.1. Cấu trúc thư mục

- **`backend/`**: Chứa ứng dụng backend (FastAPI).
  - **`api/`**: Định nghĩa các API endpoint.
  - **`core/`**: Các thành phần cốt lõi như cấu hình, bảo mật.
  - **`db/`**: Các model cơ sở dữ liệu (SQLAlchemy) và script migration (Alembic).
  - **`services/`**: Chứa logic nghiệp vụ, tương tác với các API bên ngoài (như GitHub) và cơ sở dữ liệu.
  - **`ai/`**: Chứa các mô hình AI và các dịch vụ liên quan.
- **`frontend/`**: Chứa ứng dụng frontend (React).
- **`docs/`**: Chứa tài liệu của dự án.

### 3.2. Luồng dữ liệu

1.  **Lấy dữ liệu**: Backend lấy dữ liệu commit từ kho chứa GitHub thông qua GitHub API.
2.  **Xử lý & Làm giàu dữ liệu**: `github_service` xử lý dữ liệu thô, tính toán các chỉ số bổ sung và chuẩn bị để lưu trữ.
3.  **Lưu trữ dữ liệu**: `commit_service` lưu dữ liệu commit đã xử lý vào cơ sở dữ liệu PostgreSQL.
4.  **Phân tích AI**: `han_ai_service` phân tích các commit message để phân loại chúng.
5.  **Cung cấp API**: Backend cung cấp các API endpoint để frontend có thể truy xuất dữ liệu.
6.  **Hiển thị trên Frontend**: Frontend sử dụng dữ liệu từ API để hiển thị các biểu đồ và thông tin chi tiết cho người dùng.

## 4. Công nghệ sử dụng

- **Backend**: Python, FastAPI, SQLAlchemy, PostgreSQL, Poetry.
- **Frontend**: JavaScript, React, Vite, Ant Design, Tailwind CSS.
- **AI/ML**: PyTorch, NLTK, scikit-learn.

## 5. Hướng dẫn cài đặt

Để bắt đầu, bạn cần cài đặt backend và frontend một cách riêng biệt.

### 5.1. Cài đặt Backend

1.  Di chuyển vào thư mục `backend/`.
2.  Cài đặt các gói phụ thuộc bằng `poetry install`.
3.  Tạo file `.env` và cấu hình kết nối cơ sở dữ liệu.
4.  Chạy migration để tạo các bảng trong database: `alembic upgrade head`.
5.  Khởi động server: `uvicorn main:app --reload`.

### 5.2. Cài đặt Frontend

1.  Di chuyển vào thư mục `frontend/`.
2.  Cài đặt các gói phụ thuộc bằng `npm install`.
3.  Tạo file `.env.local` và thiết lập biến `VITE_API_URL` trỏ đến địa chỉ của backend.
4.  Khởi động server: `npm run dev`.

_Để biết hướng dẫn chi tiết hơn, vui lòng tham khảo các file `README.md` trong từng thư mục._
