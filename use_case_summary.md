# Bảng Tóm tắt Use Cases - Hệ thống AI Quản lý Tiến độ Dự án

## 🎯 Tổng quan hệ thống

Hệ thống AI hỗ trợ quản lý tiến độ và phân công công việc trong dự án lập trình với 2 actor chính: **Team Leader** và **Project Manager**.

## 👥 Actors và Vai trò

| Actor               | Vai trò       | Mô tả                                                                        |
| ------------------- | ------------- | ---------------------------------------------------------------------------- |
| **Team Leader**     | Trưởng nhóm   | Quản lý trực tiếp các thành viên, phân công công việc, theo dõi tiến độ nhóm |
| **Project Manager** | Quản lý dự án | Có cái nhìn tổng quan về toàn bộ dự án, ra quyết định chiến lược             |

## 📋 Use Cases theo Actor

### 👨‍💼 Team Leader Use Cases

| ID        | Use Case            | Mô tả                                                   | Độ ưu tiên |
| --------- | ------------------- | ------------------------------------------------------- | ---------- |
| **TL-01** | Quản lý Thành viên  | Xem danh sách, trạng thái hoạt động, phát hiện inactive | Cao        |
| **TL-02** | Phân công Công việc | Nhận gợi ý AI, chỉnh sửa assignee, đồng bộ với Git      | Cao        |
| **TL-03** | Theo dõi Workload   | Xem workload từng thành viên, velocity, cân bằng tải    | Cao        |
| **TL-04** | Dashboard Nhóm      | Xem metrics nhóm, biểu đồ Gantt, heatmap                | Trung bình |
| **TL-05** | Quản lý Cảnh báo    | Nhận/xử lý cảnh báo quá tải, thiết lập ngưỡng           | Trung bình |
| **TL-06** | Đồng bộ Assignment  | Cập nhật phân công với GitHub/GitLab                    | Thấp       |

### 👔 Project Manager Use Cases

| ID        | Use Case               | Mô tả                                                      | Độ ưu tiên |
| --------- | ---------------------- | ---------------------------------------------------------- | ---------- |
| **PM-01** | Quản lý Dự án Tổng thể | Xem tổng quan tiến độ, milestone, quản lý nhiều team       | Cao        |
| **PM-02** | Báo cáo & Phân tích    | Xem báo cáo commit, xuất PDF, phân tích xu hướng           | Cao        |
| **PM-03** | Quản lý Repository     | Kết nối GitHub/GitLab, đồng bộ dữ liệu, quản lý quyền      | Cao        |
| **PM-04** | Quyết định Chiến lược  | Báo cáo hiệu suất, phân bổ tài nguyên, điều chỉnh timeline | Trung bình |
| **PM-05** | Dự đoán & Dự báo       | Phân tích xu hướng, dự đoán tiến độ, risk assessment       | Thấp       |

### 🔄 Shared Use Cases

| ID        | Use Case              | Actors | Mô tả                                             | Độ ưu tiên |
| --------- | --------------------- | ------ | ------------------------------------------------- | ---------- |
| **SH-01** | Xác thực & Phân quyền | TL, PM | Đăng nhập, quản lý profile, phân quyền            | Cao        |
| **SH-02** | AI Phân tích Commit   | TL, PM | Tự động phân loại commit, tính workload           | Cao        |
| **SH-03** | Tích hợp External     | TL, PM | Kết nối GitHub/GitLab OAuth, đồng bộ dữ liệu      | Cao        |
| **SH-04** | Dashboard Chung       | TL, PM | Hiển thị thông tin theo role, tùy chỉnh giao diện | Trung bình |
| **SH-05** | Notification System   | TL, PM | Gửi thông báo email, in-app notifications         | Thấp       |

## 🔗 Mối quan hệ Use Cases

### Include Relationships

- **TL-02 (Phân công)** includes **SH-02 (AI Phân tích)**
- **TL-01 (Quản lý thành viên)** includes **SH-02 (AI Phân tích)**
- **PM-02 (Báo cáo)** includes **SH-02 (AI Phân tích)**
- **TL-05 (Cảnh báo)** includes **SH-05 (Notification)**

### Extend Relationships

- **Gợi ý AI** extends **TL-02 (Phân công công việc)**
- **Cảnh báo tự động** extends **TL-03 (Theo dõi workload)**
- **Export PDF** extends **PM-02 (Báo cáo & phân tích)**

## 🎯 Use Cases chi tiết quan trọng

### UC-001: Phân tích Commit tự động (AI Core)

**Actors**: Team Leader, Project Manager  
**Mô tả**: AI tự động phân tích commit để phân loại công việc và đánh giá tiến độ  
**Preconditions**: Repository đã kết nối  
**Main Flow**:

1. Hệ thống thu thập commit từ GitHub/GitLab
2. AI phân loại commit (bug fix, feature, refactor, docs)
3. Tính toán workload và velocity
4. Cập nhật metrics và dashboard

### UC-002: Gợi ý Phân công Thông minh

**Actor**: Team Leader  
**Mô tả**: AI đề xuất phân công dựa trên workload và khả năng thành viên  
**Preconditions**: Có dữ liệu lịch sử commit  
**Main Flow**:

1. Team Leader yêu cầu gợi ý
2. AI phân tích workload hiện tại
3. Đề xuất assignment tối ưu
4. Team Leader xem xét và quyết định
5. Cập nhật assignment

### UC-003: Cảnh báo Quá tải

**Actors**: Team Leader, Project Manager  
**Mô tả**: Tự động phát hiện và cảnh báo workload quá cao  
**Preconditions**: Đã thiết lập ngưỡng cảnh báo  
**Main Flow**:

1. Hệ thống theo dõi workload liên tục
2. Phát hiện thành viên quá tải
3. Tạo và gửi cảnh báo
4. Đề xuất giải pháp tái phân bổ

## 🏗️ System Architecture Context

```
Frontend (React) ↔ Backend API (FastAPI) ↔ AI/ML Models ↔ Database (PostgreSQL)
                           ↕
                  GitHub/GitLab APIs
```

## 📊 Metrics & KPIs

### Team Leader Metrics:

- Workload distribution balance
- Team velocity trends
- Issue resolution time
- Member activity levels

### Project Manager Metrics:

- Overall project progress
- Cross-team performance comparison
- Resource utilization
- Timeline adherence

---

_Tài liệu này mô tả Use Cases cho hệ thống KLTN04 - AI hỗ trợ quản lý tiến độ dự án_
