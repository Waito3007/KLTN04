# TaskFlowAI - Use Case Summary (Final Version)

## 🎯 Hệ thống thực tế: TaskFlowAI

**Mô tả**: Hệ thống phân tích commit và quản lý task sử dụng AI
**Tech Stack**: FastAPI + React + PostgreSQL + GitHub API + XGBoost AI

## 📊 Data Sources (Corrected)

### ✅ Sử dụng:

- **GitHub API**: Repository, commits, issues, pull requests
- **PostgreSQL Database**: User data, task assignments, analytics
- **AI Models**: Local trained models (XGBoost, RandomForest)

### ❌ Không sử dụng:

- **GitLab**: Không có tích hợp GitLab trong hệ thống
- **External APIs khác**: Chỉ tập trung vào GitHub ecosystem

## 👥 Actors & Use Cases

### 👨‍💻 Developer (6 use cases)

- GitHub OAuth Login
- Personal dashboard & repository browsing
- Commit analysis cá nhân
- Task management cá nhân

### 👨‍💼 Team Leader (15 use cases)

- Team management & monitoring
- Kanban task board với AI suggestions
- Workload balancing & assignment
- Team analytics & insights

### 👔 Project Manager (12 use cases)

- Executive dashboard & multi-repo overview
- Advanced analytics & reporting
- Strategic planning & resource allocation
- AI-powered predictions & risk analysis

## 🔄 Key Data Flows

```
GitHub OAuth → User Authentication → Dashboard
GitHub API → Commit Data → AI Analysis → Insights
Database → Task Management → AI Assignment → Workload Optimization
```

## 🤖 AI Components

1. **XGBoost Commit Classifier** (commit_classifier_v1.joblib)

   - Phân loại commit messages
   - Multi-label classification
   - High accuracy predictions

2. **Task Assignment AI**

   - Workload balancing
   - Skill-based matching
   - Optimal resource allocation

3. **Analytics & Predictions**
   - Progress forecasting
   - Risk analysis
   - Performance trends

## 🏗️ Architecture Overview

```
React Frontend ↔ FastAPI Backend ↔ PostgreSQL Database
                     ↕
                GitHub API Only
                     ↕
              AI Models (Local)
```

## 📋 Use Case Priorities

### High Priority (Core Features):

- GitHub authentication & integration
- Commit analysis AI
- Task board management
- Basic reporting

### Medium Priority (Enhanced Features):

- Advanced analytics
- AI-powered suggestions
- Workload optimization
- Executive dashboards

### Low Priority (Nice-to-have):

- Export/import capabilities
- Advanced notification system
- Mobile responsiveness
- Custom reporting

---

_Final corrected analysis - TaskFlowAI system using GitHub + PostgreSQL only_
