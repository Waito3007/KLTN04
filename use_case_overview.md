# 🎯 USE CASE OVERVIEW - TASKFLOWAI

## Biểu đồ Use Case Tổng quan

```mermaid
graph TB
    %% Actor
    USER[👨‍💻 GitHub User]

    %% Main System Categories
    subgraph System["🎯 TaskFlowAI System"]

        subgraph AUTH["🔐 Authentication"]
            UC_AUTH[GitHub OAuth Login]
        end

        subgraph REPO["📂 Repository Management"]
            UC_REPO[Repository Operations]
        end

        subgraph AI["🤖 AI Analysis"]
            UC_AI[Commit Analysis with AI]
        end

        subgraph TASK["📋 Task Management"]
            UC_TASK[Project Task Management]
        end

        subgraph INSIGHT["💡 AI Insights"]
            UC_INSIGHT[Smart Recommendations]
        end

        subgraph REPORT["📊 Analytics & Reports"]
            UC_REPORT[Data Analytics]
        end
    end

    %% External Systems
    GitHub[🐙 GitHub API]
    Database[🗄️ PostgreSQL]

    %% User connections
    USER --> UC_AUTH
    USER --> UC_REPO
    USER --> UC_AI
    USER --> UC_TASK
    USER --> UC_INSIGHT
    USER --> UC_REPORT

    %% System connections
    UC_AUTH --> GitHub
    UC_REPO --> GitHub
    UC_AI --> GitHub
    UC_TASK --> Database
    UC_REPORT --> Database

    %% Include relationships
    UC_REPO -.->|includes| UC_AUTH
    UC_AI -.->|includes| UC_REPO
    UC_TASK -.->|includes| UC_AUTH
    UC_INSIGHT -.->|includes| UC_AI
    UC_REPORT -.->|includes| UC_AI

    classDef actor fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef auth fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef repo fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef ai fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef task fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef insight fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef report fill:#f1f8e9,stroke:#388e3c,stroke-width:2px
    classDef external fill:#ffebee,stroke:#c62828,stroke-width:2px

    class USER actor
    class AUTH,UC_AUTH auth
    class REPO,UC_REPO repo
    class AI,UC_AI ai
    class TASK,UC_TASK task
    class INSIGHT,UC_INSIGHT insight
    class REPORT,UC_REPORT report
    class GitHub,Database external
```

---

## 📋 TỔNG QUAN CÁC NHÓM USE CASE

### 🔐 **Authentication**

- GitHub OAuth login/logout
- User profile management
- Token management

### 📂 **Repository Management**

- View repositories list
- Connect/sync repositories from GitHub
- Repository details & branch management
- Repository filtering

### 🤖 **AI Analysis**

- HAN Model commit analysis
- CodeBERT embeddings
- Multi-task learning
- Commit classification & statistics

### 📋 **Task Management**

- Kanban task board
- Task CRUD operations
- Assignment management
- Priority & deadline tracking

### 💡 **AI Insights**

- Smart assignment suggestions
- Workload warnings
- Progress predictions
- Developer profiling

### 📊 **Analytics & Reports**

- Commit analysis reports
- Metrics dashboard
- Data export/import
- GitHub issues/PRs tracking

---

## 🎭 ACTOR CHARACTERISTICS

### **GitHub User** 👨‍💻

- **Access Level**: Full system access
- **Authentication**: GitHub OAuth required
- **Permissions**: No role restrictions
- **Capabilities**: All features available

---

## 🔄 SYSTEM FLOW

```
1. GitHub OAuth → 2. Repository Sync → 3. AI Analysis → 4. Task Management → 5. Insights & Reports
```

### **Core Workflow:**

1. **Login** via GitHub OAuth
2. **Connect** repositories từ GitHub
3. **Analyze** commits với AI models
4. **Manage** tasks trên Kanban board
5. **View** insights và generate reports

---

## 🏗️ TECHNICAL ARCHITECTURE

### **Frontend**: React.js + Ant Design

### **Backend**: FastAPI + Async

### **Database**: PostgreSQL + SQLAlchemy

### **AI Models**: HAN + CodeBERT + Multi-task Learning

### **Integration**: GitHub API Only

### **Authentication**: GitHub OAuth (Single-tier)
