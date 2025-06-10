# 🚀 Hướng dẫn Cài đặt và Khởi động KLTN04

## 📋 Mục lục

- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt Backend](#cài-đặt-backend)
- [Cài đặt Frontend](#cài-đặt-frontend)
- [Cài đặt AI Models](#cài-đặt-ai-models)
- [Cấu hình Database](#cấu-hình-database)
- [Khởi động hệ thống](#khởi-động-hệ-thống)
- [Kiểm tra hoạt động](#kiểm-tra-hoạt-động)
- [Troubleshooting](#troubleshooting)

---

## 💻 Yêu cầu hệ thống

### Phần mềm cần thiết:

- **Python**: 3.12 trở lên
- **Node.js**: 18.0 trở lên
- **npm**: 9.0 trở lên (hoặc yarn)
- **PostgreSQL**: 13 trở lên (tùy chọn)
- **Git**: Phiên bản mới nhất

### Kiểm tra phiên bản:

```powershell
python --version
node --version
npm --version
git --version
```

---

## 🐍 Cài đặt Backend

### Bước 1: Clone repository và di chuyển vào thư mục

```powershell
git clone <repository-url>
cd KLTN04
```

### Bước 2: Cài đặt Python dependencies

#### Tùy chọn A: Sử dụng Poetry (Khuyến nghị)

```powershell
# Cài đặt Poetry nếu chưa có
pip install poetry

# Cài đặt dependencies từ pyproject.toml
poetry install

# Kích hoạt virtual environment
poetry shell
```

#### Tùy chọn B: Sử dụng pip và venv

```powershell
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt dependencies từ requirements.txt
cd backend
pip install -r requirements.txt
```

### Bước 3: Cài đặt spaCy models

```powershell
# Tải English language model
python -m spacy download en_core_web_sm

# Tải Vietnamese model (nếu cần)
# python -m spacy download vi_core_news_sm
```

### Bước 4: Cấu hình môi trường Backend

Tạo file `.env` trong thư mục `backend/`:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/kltn04
# Hoặc sử dụng SQLite cho development:
# DATABASE_URL=sqlite:///./kltn04.db

# GitHub OAuth
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here

# Security
SECRET_KEY=your_super_secret_key_here_min_32_chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment
ENVIRONMENT=development
DEBUG=true

# Logging
LOG_LEVEL=INFO
```

### Bước 5: Setup Database (nếu sử dụng PostgreSQL)

```powershell
# Chạy migration
cd backend
alembic upgrade head
```

---

## ⚛️ Cài đặt Frontend

### Bước 1: Cài đặt Node.js dependencies

```powershell
cd frontend
npm install

# Hoặc sử dụng yarn
# yarn install
```

### Bước 2: Cấu hình môi trường Frontend

Tạo file `.env` trong thư mục `frontend/`:

```env
# API Configuration
VITE_API_URL=http://localhost:8000
VITE_API_BASE_URL=http://localhost:8000/api

# GitHub OAuth
VITE_GITHUB_CLIENT_ID=your_github_client_id_here

# Environment
VITE_NODE_ENV=development
```

---

## 🤖 Cài đặt AI Models

### Bước 1: Chuẩn bị môi trường AI

```powershell
cd backend\ai
```

### Bước 2: Tải và chuẩn bị models

```powershell
# Test commit analyzer
python test_commit_analyzer.py

# Chạy phân tích cơ bản
python simple_advanced_analysis.py
```

### Bước 3: Huấn luyện models (nếu cần)

```powershell
# Huấn luyện HAN model cho GitHub
python train_han_github.py

# Debug classification
python debug_classification_fixed.py
```

---

## 🗄️ Cấu hình Database

### PostgreSQL Setup (Khuyến nghị cho Production)

#### Bước 1: Cài đặt PostgreSQL

```powershell
# Sử dụng chocolatey (Windows)
choco install postgresql

# Hoặc tải từ: https://www.postgresql.org/download/windows/
```

#### Bước 2: Tạo database và user

```sql
-- Kết nối PostgreSQL với psql
psql -U postgres

-- Tạo database
CREATE DATABASE kltn04;

-- Tạo user
CREATE USER kltn04_user WITH PASSWORD 'your_secure_password';

-- Cấp quyền
GRANT ALL PRIVILEGES ON DATABASE kltn04 TO kltn04_user;
GRANT ALL ON SCHEMA public TO kltn04_user;

-- Thoát
\q
```

### SQLite Setup (Đơn giản cho Development)

```powershell
# SQLite sẽ tự động tạo file database khi chạy ứng dụng
# Chỉ cần cấu hình DATABASE_URL trong .env như sau:
# DATABASE_URL=sqlite:///./kltn04.db
```

---

## 🏃‍♂️ Khởi động hệ thống

### Cách 1: Khởi động từng service riêng biệt

#### Terminal 1 - Backend:

```powershell
cd backend
# Kích hoạt virtual environment
poetry shell
# Hoặc: .\venv\Scripts\Activate.ps1

# Khởi động server
python main.py

# Hoặc sử dụng uvicorn trực tiếp:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2 - Frontend:

```powershell
cd frontend
npm run dev

# Hoặc sử dụng yarn:
# yarn dev
```

#### Terminal 3 - AI Processing (Tùy chọn):

```powershell
cd backend\ai
python simple_advanced_analysis.py
```

### Cách 2: Script khởi động nhanh

Tạo file `start.ps1` trong thư mục gốc:

```powershell
# start.ps1
Write-Host "🚀 Khởi động KLTN04..." -ForegroundColor Green

# Khởi động Backend
Write-Host "🐍 Khởi động Backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; poetry shell; python main.py"

# Đợi backend khởi động
Start-Sleep -Seconds 5

# Khởi động Frontend
Write-Host "⚛️ Khởi động Frontend..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "✅ Hệ thống đã khởi động!" -ForegroundColor Green
Write-Host "🌐 Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
```

Chạy script:

```powershell
.\start.ps1
```

---

## ✅ Kiểm tra hoạt động

### 1. Kiểm tra Backend

```powershell
# Test API endpoint
curl http://localhost:8000/

# Hoặc sử dụng PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get
```

### 2. Kiểm tra Frontend

Truy cập: http://localhost:5173

### 3. Kiểm tra API Documentation

Truy cập: http://localhost:8000/docs

### 4. Kiểm tra Database connection

```powershell
cd backend
python -c "from core.database import engine; print('Database connection: OK')"
```

---

## 🔧 GitHub OAuth Setup

### Bước 1: Tạo GitHub OAuth App

1. Truy cập [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Điền thông tin:
   - **Application name**: KLTN04 Local Dev
   - **Homepage URL**: http://localhost:5173
   - **Authorization callback URL**: http://localhost:8000/auth/github/callback

### Bước 2: Lấy Client ID và Secret

1. Copy **Client ID** và **Client Secret**
2. Thêm vào file `.env` của backend và frontend

---

## 🌐 URLs quan trọng

Sau khi khởi động thành công:

| Service            | URL                         | Mô tả                         |
| ------------------ | --------------------------- | ----------------------------- |
| Frontend           | http://localhost:5173       | Giao diện người dùng          |
| Backend API        | http://localhost:8000       | REST API endpoints            |
| API Docs (Swagger) | http://localhost:8000/docs  | Interactive API documentation |
| ReDoc              | http://localhost:8000/redoc | Alternative API documentation |

---

## 🐛 Troubleshooting

### Lỗi Python Dependencies

```powershell
# Cập nhật pip
python -m pip install --upgrade pip

# Cài lại dependencies
pip install -r backend\requirements.txt --force-reinstall

# Xóa cache pip
pip cache purge
```

### Lỗi spaCy Model

```powershell
# Cài lại model
python -m spacy download en_core_web_sm --force

# Kiểm tra model đã cài
python -m spacy info en_core_web_sm
```

### Lỗi Database Connection

```powershell
# Kiểm tra PostgreSQL service
Get-Service postgresql*

# Khởi động PostgreSQL service (nếu cần)
Start-Service postgresql-x64-13
```

### Lỗi Frontend Build

```powershell
# Xóa node_modules và cài lại
cd frontend
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json -ErrorAction SilentlyContinue
npm install
```

### Lỗi Port đã được sử dụng

```powershell
# Kiểm tra port 8000
netstat -ano | findstr :8000

# Kiểm tra port 5173
netstat -ano | findstr :5173

# Kill process nếu cần (thay PID bằng số thực tế)
taskkill /PID <PID> /F
```

### Lỗi CORS

Kiểm tra file `backend\core\config.py`:

```python
# Đảm bảo frontend URL trong allowed_origins
allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"]
```

### Debug Commands

```powershell
# Kiểm tra Python packages
pip list

# Kiểm tra Node packages
npm list

# Xem logs backend (nếu có)
Get-Content backend\logs\app.log -Tail 50

# Test database connection
cd backend
python -c "from db.database import get_database; print('DB OK')"
```

---

## 📦 Build cho Production

### Backend

```powershell
cd backend
pip freeze > requirements-prod.txt
```

### Frontend

```powershell
cd frontend
npm run build
npm run preview
```

---

## 🔄 Cập nhật Dependencies

### Backend

```powershell
# Sử dụng Poetry
poetry update

# Hoặc pip
pip install --upgrade -r backend\requirements.txt
```

### Frontend

```powershell
cd frontend
npm update
```

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra log files trong `backend/logs/`
2. Xem console errors trong browser (F12)
3. Chạy các debug commands ở trên
4. Tạo issue trên GitHub repository

---

**🎉 Chúc bạn cài đặt thành công!**
