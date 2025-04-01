# KLTN04 - Ứng dụng AI hỗ trợ quản lý tiến độ và phân công công việc trong dự án lập trình

## 📅 TỔNG QUAN KẾ HOẠCH
**Mục tiêu :**
- Tự động phân tích tiến độ dự án lập trình
- Đề xuất công việc tiếp theo một cách thông minh
- Hỗ trợ team leader quản lý workload hiệu quả
- Tạo báo cáo và cảnh báo tự động
- Tích hợp với hệ thống quản lý nhóm


##  CÁC GIAI ĐOẠN PHÁT TRIỂN

### 1. THIẾT LẬP CƠ BẢN (7/4 - 28/4)
**Tuần 1:**
- Setup FastAPI + React
- Kết nối GitHub API (OAuth)

**Tuần 2:**
- Thiết kế database PostgreSQL
- Lưu trữ commit/PR data

**Tuần 2.5:**
- Tích hợp GitLab API
- Chuẩn hóa data model

### 2. PHÁT TRIỂN AI CORE (29/4 - 26/5)
**Tuần 3:**
- Phân loại commit bằng spaCy
- Phát hiện thành viên inactive

**Tuần 4-5:**
- Huấn luyện model XGBoost
- Tính toán workload/velocity

**Tuần 6:**
- Triển khai API gợi ý phân công
- Hệ thống cảnh báo quá tải

### 3. TÍCH HỢP DASHBOARD (27/5 - 23/6)
**Tuần 7:**
- Dashboard hiển thị metrics
- Biểu đồ Gantt và heatmap

**Tuần 8-9:**
- Chức năng chỉnh sửa assignee
- Đồng bộ ngược với GitHub

**Tuần 10:**
- Xuất báo cáo PDF
- Tích hợp Slack alerts

### 4. HOÀN THIỆN (24/6 - 7/7)
**Tuần 11:**
- Kiểm thử hiệu năng
- A/B testing so sánh

**Tuần 12:**
- Viết báo cáo kỹ thuật
- Chuẩn bị demo sản phẩm

## 🛠 CÔNG NGHỆ CHÍNH
**Frontend:**
- React, Ant Design, D3.js

**Backend:**
- FastAPI, PostgreSQL

**AI/ML:**
- spaCy, XGBoost, SHAP

**DevOps:**
- Docker, GitHub Actions
  
