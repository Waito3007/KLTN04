# Use Case Analysis - TaskFlowAI System

## 🎯 Tổng quan hệ thống thực tế

**TaskFlowAI** là hệ thống phân tích commit và quản lý task sử dụng AI, được phát triển với:

- **Backend**: FastAPI (Python) với HAN (Hierarchical Attention Network)
- **Frontend**: React.js với Ant Design
- **Database**: PostgreSQL với SQLAlchemy
- **Integration**: GitHub API (OAuth + REST API)
- **AI Features**: HAN multi-task learning, CodeBERT embeddings, commit classification

## 👥 Actors trong hệ thống

| Actor               | Mô tả                       | Vai trò chính                                  |
| ------------------- | --------------------------- | ---------------------------------------------- |
| **Developer**       | Lập trình viên thông thường | Xem commit analysis, tự quản lý task cá nhân   |
| **Team Leader**     | Trưởng nhóm phát triển      | Quản lý team, phân công task, theo dõi tiến độ |
| **Project Manager** | Quản lý dự án               | Oversight tổng thể, báo cáo, ra quyết định     |

## 🏗️ Kiến trúc hệ thống thực tế

```
Frontend (React + Ant Design)
├── Login/Auth pages
├── Dashboard with widgets
├── Repository management
├── Commit analysis views
├── Task Board (Kanban)
└── AI Insights widgets

Backend (FastAPI)
├── OAuth authentication (/api/login, /auth/callback)
├── GitHub integration (/api/github/*)
├── Commit analysis (/api/commits/analyze-*)
├── AI services (/ai/analyze-commits, /ai/assign-tasks)
└── Repository management

AI Models
├── HAN (Hierarchical Attention Network) - han_multitask.pth
├── CodeBERT Embeddings (microsoft/codebert-base)
├── Multi-task Learning (commit classification + sentiment + purpose)
├── Text Processor & Embedding Loader
└── Legacy models (commit_classifier_v1.joblib, XGBoost)

Database (PostgreSQL)
├── users (GitHub OAuth data)
├── repositories (connected repos)
├── commits (commit analysis data)
├── assignments (task assignments)
├── branches, issues, pull_requests
```

## 📋 Use Cases Chi tiết theo Actor

### 👨‍💻 Developer (Lập trình viên)

| ID         | Use Case            | Mô tả                            | Endpoint                        | Component         |
| ---------- | ------------------- | -------------------------------- | ------------------------------- | ----------------- |
| **DEV-01** | GitHub OAuth Login  | Đăng nhập qua GitHub OAuth       | `/api/login`, `/auth/callback`  | `Login.jsx`       |
| **DEV-02** | Xem Dashboard       | Dashboard cá nhân với overview   | `/dashboard`                    | `Dashboard.jsx`   |
| **DEV-03** | Xem Repository List | Danh sách repo có quyền truy cập | `/api/github/repos`             | `RepoList.jsx`    |
| **DEV-04** | Chi tiết Repository | Xem thông tin chi tiết repo      | `/repo/:owner/:repo`            | `RepoDetails.jsx` |
| **DEV-05** | Xem Commit Analysis | Phân tích commit của bản thân    | `/api/commits/analyze-github/*` | `CommitTable.jsx` |
| **DEV-06** | Tự quản lý Task     | Xem và cập nhật task cá nhân     | `TaskBoard`                     | `TaskBoard.jsx`   |

### 👨‍💼 Team Leader (Trưởng nhóm)

| ID        | Use Case             | Mô tả                                | Endpoint                 | Component                 |
| --------- | -------------------- | ------------------------------------ | ------------------------ | ------------------------- |
| **TL-01** | Dashboard Team       | Dashboard với metrics team           | `/dashboard`             | `Dashboard.jsx` + widgets |
| **TL-02** | Kanban Task Board    | Quản lý task board của team          | `TaskBoard`              | `TaskBoard.jsx`           |
| **TL-03** | Phân công Task       | Assign/reassign tasks cho thành viên | `/ai/assign-tasks`       | Task management           |
| **TL-04** | AI Insight Widget    | Xem gợi ý và cảnh báo từ AI          | AI widgets               | `AIInsightWidget.jsx`     |
| **TL-05** | Quản lý Repository   | Kết nối và quản lý repo team         | `/api/github/*`          | Repository management     |
| **TL-06** | Commit Analysis Team | Phân tích commit của cả team         | `/api/commits/analyze-*` | Commit analytics          |
| **TL-07** | Workload Monitoring  | Theo dõi workload thành viên         | Analytics                | Workload widgets          |
| **TL-08** | Branch Management    | Quản lý branch và merge              | Branch service           | Branch components         |

### 👔 Project Manager (Quản lý dự án)

| ID        | Use Case                | Mô tả                      | Endpoint                 | Component           |
| --------- | ----------------------- | -------------------------- | ------------------------ | ------------------- |
| **PM-01** | Executive Dashboard     | Dashboard tổng quan dự án  | `/dashboard`             | Executive view      |
| **PM-02** | Multi-repo Overview     | Quản lý nhiều repository   | Repository service       | Multi-repo view     |
| **PM-03** | Commit Analytics Report | Báo cáo phân tích commit   | `/api/commits/analyze-*` | Analytics reports   |
| **PM-04** | AI Predictions          | Dự đoán tiến độ và rủi ro  | AI prediction service    | Prediction widgets  |
| **PM-05** | Resource Allocation     | Phân bổ tài nguyên optimal | AI assignment            | Resource management |
| **PM-06** | Progress Tracking       | Theo dõi tiến độ milestone | Project tracking         | Progress dashboard  |
| **PM-07** | Export Reports          | Xuất báo cáo và metrics    | Export service           | Report generator    |

## 🤖 AI Features trong hệ thống

### HAN (Hierarchical Attention Network)

```python
# Hierarchical Attention Network cho commit analysis
- Multi-task learning: purpose, sentiment, commit_type
- CodeBERT embeddings (microsoft/codebert-base)
- Attention mechanism tại word và sentence level
- PyTorch implementation với custom training
```

### CodeBERT Integration

```python
# EmbeddingLoader với CodeBERT
- Pre-trained microsoft/codebert-base model
- Transformers integration
- Word và document-level embeddings
- Cached embedding computation
```

### Multi-task Learning System

```python
# MultiTaskTrainer
- Purpose classification (9 categories)
- Sentiment analysis (positive/neutral/negative)
- Commit type detection
- Uncertainty weighting loss
```

## 🔄 User Flows chính

### 1. Developer Daily Flow

```
Login → Dashboard → Select Repo → View Commits → Check Tasks → Update Status
```

### 2. Team Leader Management Flow

```
Login → Team Dashboard → Review AI Insights → Assign Tasks → Monitor Progress → Adjust Assignments
```

### 3. Project Manager Strategic Flow

```
Login → Executive Dashboard → Multi-repo View → AI Predictions → Resource Decisions → Export Reports
```

## 🎨 Frontend Components Structure

```
src/
├── pages/
│   ├── Login.jsx (GitHub OAuth)
│   ├── Dashboard.jsx (Main dashboard)
│   ├── RepoDetails.jsx (Repository details)
│   └── AuthSuccess.jsx (OAuth callback)
├── components/
│   ├── Dashboard/
│   │   ├── OverviewCard.jsx (Metrics cards)
│   │   ├── AIInsightWidget.jsx (AI suggestions)
│   │   ├── TaskBoard.jsx (Kanban board)
│   │   └── RepoListFilter.jsx (Filter repos)
│   ├── repo/
│   │   └── RepoList.jsx (Repository list)
│   ├── commits/
│   │   └── CommitTable.jsx (Commit analysis)
│   └── Branchs/
```

## 🔧 Backend Services Structure

```
backend/
├── api/routes/
│   ├── auth.py (GitHub OAuth)
│   ├── github.py (GitHub API integration)
│   ├── commit_routes.py (Commit analysis)
│   └── ai_suggestions.py (AI endpoints)
├── services/
│   ├── ai_service.py (AI model interface)
│   ├── model_loader.py (Model management)
│   ├── github_service.py (GitHub API)
│   ├── commit_service.py (Commit processing)
│   └── repo_service.py (Repository management)
├── models/
│   ├── commit_classifier_v1.joblib (XGBoost model)
│   └── AI model files
└── db/models/
    ├── users.py (User schema)
    ├── commits.py (Commit schema)
    ├── repositories.py (Repo schema)
    └── assignments.py (Task schema)
```

## 📊 Database Schema

```sql
-- Core tables thực tế
users (id, github_id, github_username, email, avatar_url)
repositories (id, owner, name, github_id, access_token)
commits (id, sha, message, author_name, insertions, deletions, repo_id)
assignments (id, task_name, description, is_completed, user_id)
branches (id, name, repo_id)
issues (id, title, state, repo_id)
pull_requests (id, title, state, repo_id)
```

## 🚀 Key Differentiators

1. **AI-Powered**: HAN (Hierarchical Attention) + CodeBERT với multi-task learning
2. **Real GitHub Integration**: OAuth + API integration thực tế
3. **Modern UI**: React + Ant Design responsive
4. **Scalable Architecture**: FastAPI + PostgreSQL
5. **Task Management**: Kanban board với drag-drop
6. **Advanced NLP**: Deep learning models cho commit analysis

---

_Phân tích dựa trên code thực tế của hệ thống TaskFlowAI KLTN04_
