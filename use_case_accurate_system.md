# 🎯 USE CASE DIAGRAM - TASKFLOWAI SYSTEM

## Phân tích chính xác dựa trên codebase thực tế

---

## 🎭 ACTOR THỰC TẾ

### **GitHub User** 👨‍💻

- **Định nghĩa**: Bất kỳ người dùng nào đăng nhập qua GitHub OAuth
- **Quyền hạn**: Full access to all system features (không có role restriction)
- **Authentication**: GitHub OAuth với scopes: `read:user user:email repo`
- **Đặc điểm**:
  - Không có phân biệt role (Developer/Manager)
  - Tất cả user đều có quyền như nhau
  - Chỉ cần có GitHub account và access token hợp lệ

---

## 📋 CÁC USE CASE THỰC TẾ

### **1. AUTHENTICATION & PROFILE** 🔐

#### UC01: GitHub OAuth Login

- **Actor**: GitHub User
- **Mô tả**: Đăng nhập qua GitHub OAuth
- **Flow chính**:
  1. User click "Login with GitHub"
  2. Redirect to GitHub authorization
  3. GitHub callback với access token
  4. Save user info vào database (users table)
  5. Redirect to Dashboard

#### UC02: Quản lý Profile User

- **Actor**: GitHub User
- **Mô tả**: Xem thông tin profile từ GitHub
- **Thông tin**: github_username, email, avatar_url, bio, location, company

#### UC03: Dashboard Tổng quan

- **Actor**: GitHub User
- **Mô tả**: Xem dashboard với repositories và AI insights
- **Include**: UC21 (Metrics Overview), UC16 (AI Insight Widget)

---

### **2. REPOSITORY MANAGEMENT** 📂

#### UC04: Xem Danh sách Repository

- **Actor**: GitHub User
- **Mô tả**: Xem repos từ database hoặc GitHub API
- **Data Sources**: Database (primary) → GitHub API (fallback)

#### UC05: Kết nối Repository

- **Actor**: GitHub User
- **Mô tả**: Sync repositories từ GitHub về database
- **Note**: Chỉ repos mà user có quyền access

#### UC06: Chi tiết Repository

- **Actor**: GitHub User
- **Mô tả**: Xem thông tin repo: commits, branches, issues, PRs

#### UC07: Quản lý Branch

- **Actor**: GitHub User
- **Mô tả**: Xem danh sách branches từ database/GitHub

---

### **3. COMMIT ANALYSIS (AI CORE)** 🤖

#### UC08: Phân tích Commit AI

- **Actor**: GitHub User
- **Mô tả**: Sử dụng HAN + CodeBERT để phân tích commits
- **AI Models**:
  - HAN (Hierarchical Attention Network): `han_multitask.pth`
  - CodeBERT: `microsoft/codebert-base`
  - Multi-task Learning framework

#### UC09: Phân loại Commit Message

- **Actor**: GitHub User
- **Mô tả**: AI categorization của commit messages
- **Categories**: feat, fix, docs, style, refactor, test, chore

#### UC10: Thống kê Commit

- **Actor**: GitHub User
- **Mô tả**: Analytics về commit patterns, frequency, impact

#### UC11: Tìm kiếm Commit

- **Actor**: GitHub User
- **Mô tả**: Search và filter commits với various criteria

---

### **4. TASK MANAGEMENT** 📋

#### UC12: Kanban Task Board

- **Actor**: GitHub User
- **Mô tả**: Kanban board với TODO/IN_PROGRESS/DONE columns
- **Tech**: React DnD, Ant Design

#### UC13: Quản lý Assignment

- **Actor**: GitHub User
- **Mô tả**: Assign tasks cho collaborators
- **Database**: project_tasks table

#### UC14: Tạo/Chỉnh sửa Task

- **Actor**: GitHub User
- **Mô tả**: CRUD operations cho tasks
- **Fields**: title, description, assignee, priority, status, due_date

#### UC15: Phân công Task

- **Actor**: GitHub User
- **Mô tả**: Assign tasks dựa trên AI suggestions hoặc manual

---

### **5. AI INSIGHTS & SUGGESTIONS** 💡

#### UC16: AI Insight Widget

- **Actor**: GitHub User
- **Mô tả**: Dashboard widget hiển thị AI insights
- **Include**: UC18 (Workload warnings)

#### UC17: Gợi ý Phân công Thông minh

- **Actor**: GitHub User
- **Mô tả**: AI-powered task assignment recommendations
- **Service**: HANAIService

#### UC18: Cảnh báo Workload

- **Actor**: GitHub User
- **Mô tả**: Detect overloaded team members

#### UC19: Dự đoán Tiến độ

- **Actor**: GitHub User
- **Mô tả**: Project timeline predictions dựa trên historical data

---

### **6. REPORTING & ANALYTICS** 📊

#### UC20: Báo cáo Commit Analysis

- **Actor**: GitHub User
- **Mô tả**: Detailed reports từ AI analysis
- **Include**: UC08 (AI Analysis)

#### UC21: Metrics Overview Card

- **Actor**: GitHub User
- **Mô tả**: Dashboard cards với key metrics

#### UC22: Export/Import Data

- **Actor**: GitHub User
- **Mô tả**: Export reports, import project data

#### UC23: Theo dõi Issues/PRs

- **Actor**: GitHub User
- **Mô tả**: Monitor GitHub issues và pull requests

---

### **7. ADVANCED FEATURES** 🔧

#### UC24: Filter Repository

- **Actor**: GitHub User
- **Mô tả**: Advanced filtering options cho repo list

#### UC25: Responsive UI

- **Actor**: GitHub User
- **Mô tả**: Mobile-friendly interface với Ant Design

#### UC26: Notification System

- **Actor**: GitHub User
- **Mô tả**: Real-time notifications cho task updates

---

### **8. AI MODELS (CORE ENGINE)** 🧠

#### UC27: HAN Model (Hierarchical Attention)

- **Actor**: GitHub User (thông qua UC08)
- **Mô tả**: Core AI model cho commit analysis
- **File**: `han_multitask.pth`

#### UC28: CodeBERT Embeddings

- **Actor**: GitHub User (thông qua UC09)
- **Mô tả**: Code understanding với Microsoft CodeBERT
- **Model**: `microsoft/codebert-base`

#### UC29: Multi-task Learning

- **Actor**: GitHub User (thông qua UC17)
- **Mô tả**: Advanced ML cho multiple prediction tasks

---

## 🔗 EXTERNAL SYSTEMS

### **GitHub API** 🐙

- OAuth authentication
- Repository data
- Commit information
- Issues & Pull Requests
- Collaborator information

### **PostgreSQL Database** 🗄️

- Users management
- Repository caching
- Task storage
- Commit analysis results
- Project assignments

---

## 🎯 SYSTEM CHARACTERISTICS

### **Authentication Model**

- Single-tier: Tất cả GitHub users có same privileges
- No role-based access control
- Token-based authentication via GitHub OAuth

### **Data Architecture**

- Database-first với GitHub API fallback
- Intelligent caching strategy
- Real-time sync capabilities

### **AI Integration**

- HAN + CodeBERT pipeline
- Multi-task learning framework
- Real-time inference capabilities

### **UI/UX Pattern**

- Single Page Application (React)
- Responsive design (Ant Design)
- Real-time updates với WebSocket potential

---

## ✅ VERIFIED AGAINST CODEBASE

✅ **Authentication**: GitHub OAuth only, no role differentiation  
✅ **Database Models**: users, repositories, project_tasks, commits, etc.  
✅ **AI Models**: HAN, CodeBERT confirmed in codebase  
✅ **API Routes**: FastAPI với authentication dependencies  
✅ **Frontend**: React/Ant Design với full feature access for all users  
✅ **No GitLab**: Only GitHub integration confirmed
