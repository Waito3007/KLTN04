# Khóa luận tốt nghiệp: Ứng dụng AI hỗ trợ quản lý tiến độ và phân công công việc trong dự án lập trình

[English Version](README_EN.md)

Dự án này là một hệ thống thông minh sử dụng Trí tuệ nhân tạo (AI) để phân tích dữ liệu từ các kho chứa mã nguồn Git, nhằm hỗ trợ các nhà quản lý dự án trong việc theo dõi tiến độ, đánh giá rủi ro và đưa ra các gợi ý phân công công việc một cách hiệu quả.

---

## 🎯 Giới thiệu

Trong các dự án phát triển phần mềm, việc quản lý và phân công công việc một cách tối ưu là yếu tố then chốt dẫn đến thành công. Hệ thống này được xây dựng để giải quyết các thách thức đó bằng cách:

- **Tự động hóa** việc phân tích các commit và hoạt động trên kho mã nguồn.
- **Cung cấp số liệu trực quan** về hiệu suất, đóng góp và các lĩnh vực chuyên môn của từng thành viên.
- **Sử dụng mô hình AI** để phân loại commit, đánh giá mức độ phức tạp và rủi ro, từ đó gợi ý người thực hiện phù hợp cho các công việc (task/issue).

---

## ✨ Tính năng chính

- **Dashboard trực quan**: Hiển thị tổng quan về sức khỏe dự án, hoạt động gần đây, và các chỉ số quan trọng.
- **Phân tích Commit thông minh**: Tự động phân loại commit (tính năng mới, sửa lỗi, tái cấu trúc,...) và đánh giá độ phức tạp bằng mô hình AI.
- **Tích hợp GitHub**: Đồng bộ hóa dữ liệu repositories, commits, issues, và branches từ GitHub.
- **Phân tích thành viên**: Xây dựng hồ sơ năng lực (skill profile) cho từng thành viên dựa trên lịch sử đóng góp.
- **Đánh giá rủi ro**: Phân tích các commit và thay đổi trong mã nguồn để cảnh báo sớm các rủi ro tiềm ẩn.
- **Gợi ý phân công công việc**: Dựa trên hồ sơ năng lực và nội dung công việc, hệ thống gợi ý thành viên phù hợp nhất để thực hiện.
- **Quản lý dự án và kho chứa**: Cho phép thêm, quản lý và theo dõi nhiều dự án và repositories.

---

## 🛠️ Công nghệ sử dụng

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React"/>
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite"/>
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

- **Backend**:
  - **Ngôn ngữ**: Python 3.9+
  - **Framework**: FastAPI
  - **ORM**: SQLAlchemy với Alembic để quản lý migrations.
  - **Quản lý môi trường**: Poetry
- **Frontend**:
  - **Framework**: React.js
  - **Build tool**: Vite
  - **Ngôn ngữ**: JavaScript, JSX
  - **Styling**: CSS, có thể tích hợp các thư viện như Material-UI hoặc Ant Design.
- **AI & Machine Learning**:
  - **Thư viện**: PyTorch/TensorFlow, Scikit-learn, Pandas, NLTK.
  - **Mô hình**:
    - **HAN (Hierarchical Attention Network)**: Dùng để phân loại văn bản (commit messages).
    - **MultiFusion Model**: Mô hình tùy chỉnh kết hợp nhiều nguồn dữ liệu (commit message, file thay đổi,...) để phân tích và đưa ra dự đoán.
- **Cơ sở dữ liệu**: PostgreSQL
- **CI/CD & Deployment**: Docker (dự kiến)

---

## 🏗️ Kiến trúc hệ thống

Dự án được xây dựng theo kiến trúc Monorepo, bao gồm 2 thành phần chính:

- **`backend/`**: Chứa toàn bộ logic nghiệp vụ, API endpoints, xử lý AI và tương tác với cơ sở dữ liệu.
- **`frontend/`**: Giao diện người dùng được xây dựng bằng React để tương tác với API từ backend.

### **Luồng hoạt động cơ bản**

1. Người dùng đăng nhập và thêm một repository từ GitHub vào hệ thống.
2. Backend thực hiện quá trình đồng bộ hóa dữ liệu (commits, issues, contributors...) từ GitHub API.
3. Dữ liệu được lưu vào CSDL PostgreSQL.
4. Các mô hình AI xử lý dữ liệu đã đồng bộ (ví dụ: phân loại commit) và lưu lại kết quả.
5. Frontend gọi API từ backend để hiển thị các thông tin phân tích trên giao diện cho người dùng.

---

## 🚀 Hướng dẫn Cài đặt và Chạy dự án

### **Yêu cầu**

- Python 3.9+ và Poetry
- Node.js 18+ và npm/yarn
- PostgreSQL Server

### **1. Cài đặt Backend**

```bash
# 1. Di chuyển vào thư mục backend
cd backend

# 2. Cài đặt các dependencies bằng Poetry
poetry install

# 3. Cấu hình môi trường
# Tạo file .env và cấu hình các biến môi trường cần thiết
# (DATABASE_URL, GITHUB_TOKEN,...) theo file .env.example (nếu có)
cp .env.example .env
# nano .env

# 4. Chạy database migrations
alembic upgrade head

# 5. Khởi động server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### **2. Cài đặt Frontend**

```bash
# 1. Mở một terminal khác, di chuyển vào thư mục frontend
cd frontend

# 2. Cài đặt các dependencies
npm install

# 3. Cấu hình API endpoint
# Mở file src/config.js và đảm bảo địa chỉ API trỏ về backend (http://localhost:8000)

# 4. Khởi động development server
npm run dev
```

Sau khi hoàn tất, truy cập `http://localhost:5173` (hoặc cổng mà Vite cung cấp) trên trình duyệt để sử dụng ứng dụng.

---

## 📂 Cấu trúc thư mục

```plaintext
.
├── backend/        # Source code của Backend (FastAPI)
│   ├── ai/         # Các logic về mô hình AI, training, prediction
│   ├── api/        # Định nghĩa các API endpoints (routers)
│   ├── core/       # Cấu hình chung, security, middleware
│   ├── db/         # Thiết lập database, models
│   ├── schemas/    # Pydantic schemas (data validation)
│   ├── services/   # Nơi chứa logic nghiệp vụ chính
│   └── main.py     # Entrypoint của ứng dụng Backend
├── frontend/       # Source code của Frontend (React)
│   ├── public/
│   └── src/
│       ├── api/
│       ├── components/
│       ├── features/
│       ├── pages/
│       └── App.jsx
├── docs/           # Các tài liệu của dự án
└── README.md       # File bạn đang đọc
```

---

## 📌 Thông tin dự án

- **Sinh viên thực hiện**: Vũ Phan Hoài Sang, Lê Trọng Nghĩa
- **Giảng viên hướng dẫn**: ThS. Đặng Thị Kim Giao
- **Trường**: TRƯỜNG ĐẠI HỌC NGOẠI NGỮ - TIN HỌC TP.HỒ CHÍ MINH (HUFLIT)
- **Năm**: 2025
