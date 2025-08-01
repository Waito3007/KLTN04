# Use Case Diagram - TaskFlowAI: Hệ thống AI Phân tích Commit & Quản lý Task

## Actor

### **GitHub User** 👨‍💻

**Định nghĩa**: Bất kỳ người dùng nào đăng nhập qua GitHub OAuth  
**Đặc điểm**: Tất cả user có quyền truy cập đầy đủ vào system (không có phân quyền role)

## Use Cases (Áp dụng cho GitHub User)

### 1. **Authentication & Core**

- Xem danh sách thành viên và trạng thái hoạt động
- Phát hiện thành viên inactive
- Quản lý profile và thông tin cá nhân
- View team members và collaborators

### 2. **Repository Management**

- Xem danh sách repositories
- Kết nối và sync repositories từ GitHub
- Xem chi tiết repository (commits, branches, issues)
- Quản lý branches
- Filter repositories

### 3. **Task Management**

- Tạo và chỉnh sửa tasks
- Phân công tasks cho team members
- Quản lý Kanban task board (TODO/IN_PROGRESS/DONE)
- Quản lý assignments
- Set priorities và due dates

### 4. **Commit Analysis với AI**

- Phân tích commits bằng HAN Model
- Phân loại commit messages với CodeBERT
- Thống kê commit patterns
- Tìm kiếm và filter commits
- Multi-task learning insights

### 5. **AI Insights & Suggestions**

- Xem AI insight widgets
- Nhận gợi ý phân công thông minh từ AI
- Cảnh báo workload quá tải
- Dự đoán tiến độ dự án
- AI-powered recommendations

### 6. **Reporting & Analytics**

- Xem báo cáo commit analysis
- Metrics overview cards
- Export/Import project data
- Theo dõi GitHub issues và pull requests
- Dashboard analytics

### 7. **Advanced Features**

- Responsive UI cho mobile/desktop
- Notification system
- Real-time updates
- Multi-language support potential

### 8. **Authentication & Configuration**

- GitHub OAuth login
- Quản lý profile cá nhân
- Dashboard customization
- System configuration

## Mô tả chi tiết Use Cases

### UC01: GitHub OAuth Authentication

**Actor**: Team Leader/Project Manager
**Mô tả**: Hệ thống sử dụng HAN (Hierarchical Attention Network) và CodeBERT để phân tích commit
**Precondition**: Repository đã được kết nối via GitHub OAuth
**Flow**:

**Actor**: GitHub User
**Mô tả**: Đăng nhập vào hệ thống thông qua GitHub OAuth
**Precondition**: User có GitHub account hợp lệ
**Flow**:

1. User click "Login with GitHub" button
2. Redirect tới GitHub OAuth authorization page
3. User authorize application với scopes: read:user, user:email, repo
4. GitHub callback với authorization code
5. Exchange code cho access token
6. Save user information vào database
7. Redirect về Dashboard

### UC02: AI Commit Analysis với HAN + CodeBERT

**Actor**: GitHub User  
**Mô tả**: Phân tích commits sử dụng HAN và CodeBERT models
**Precondition**: Có repository data và trained models
**Flow**:

1. User select repository để phân tích
2. Hệ thống load commits từ database/GitHub API
3. HAN model phân loại commit categories (feat, fix, docs, etc.)
4. CodeBERT embeddings phân tích code semantics
5. Multi-task learning tạo insights và statistics
6. Display results trên Dashboard và commit table

### UC03: Task Management với Kanban Board

**Actor**: GitHub User
**Mô tả**: Quản lý tasks thông qua Kanban interface
**Precondition**: User đã select repository
**Flow**:

1. User access Task Board tab
2. System load existing tasks từ project_tasks table
3. Display Kanban với columns: TODO, IN_PROGRESS, DONE
4. User có thể drag & drop tasks giữa columns
5. Create new tasks với form: title, description, assignee, priority, due_date
6. System sync tasks vào database

### UC04: AI-powered Smart Assignment

**Actor**: GitHub User
**Mô tả**: AI gợi ý phân công tasks dựa trên developer profiles
**Precondition**: Có commit history và task data
**Flow**:

1. User create/edit task và click "AI Suggest Assignee"
2. HANAIService analyze developer profiles từ commit patterns
3. Calculate match scores based on task type và developer specialization
4. Present recommended assignee với confidence score và reasoning
5. User accept/modify suggestion và save task

## System Architecture

### **Core Components:**

- **Frontend**: React.js với Ant Design (Dashboard, Kanban, Analytics)
- **Backend**: FastAPI với async/await support
- **Database**: PostgreSQL với SQLAlchemy ORM
- **AI Engine**: HAN + CodeBERT + Multi-task Learning
- **Integration**: GitHub API Only (OAuth + REST API)

### **AI Models:**

- **HAN Model**: `han_multitask.pth` - Hierarchical Attention Network
- **CodeBERT**: `microsoft/codebert-base` - Code understanding
- **Multi-task Learning**: Task assignment recommendations

### **Authentication:**

- **Single-tier Model**: Tất cả GitHub users có same privileges
- **No Role-based Access Control**: Không phân biệt Developer/Manager
- **GitHub OAuth**: Scopes: `read:user user:email repo`

### **Data Sources:**

- **Primary**: PostgreSQL database (repositories, tasks, users, commits)
- **Fallback**: GitHub API real-time data
- **No GitLab Integration**: Chỉ GitHub only

### AI Models:

- **HAN Model**: han_multitask.pth (Purpose + Sentiment + Type classification)
- **CodeBERT**: microsoft/codebert-base (Code-aware embeddings)
- **Multi-task Trainer**: Uncertainty weighting loss optimization

## External Dependencies

- **GitHub API**: Repository data, commits, issues, pull requests
- **PostgreSQL**: Persistent storage cho users, assignments, analytics
- **Model Storage**: Local AI models và cached embeddings
