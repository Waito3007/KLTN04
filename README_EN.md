# Graduation Thesis: AI Application to Support Progress Management and Task Assignment in Software Projects

[Phiên bản Tiếng Việt](README.md)

This project is an intelligent system utilizing Artificial Intelligence (AI) to analyze data from Git repositories, aiming to assist project managers in tracking progress, assessing risks, and providing effective task assignment suggestions.

---

## 🎯 Introduction

In software development projects, optimal management and task assignment are key factors leading to success. This system is designed to address these challenges by:

- **Automating** the analysis of commits and activities in the repository.
- **Providing visual metrics** on performance, contributions, and areas of expertise for each member.
- **Using AI models** to classify commits, assess complexity and risks, and suggest suitable individuals for tasks/issues.

---

## ✨ Key Features

- **Interactive Dashboard**: Displays an overview of project health, recent activities, and key metrics.
- **Intelligent Commit Analysis**: Automatically classifies commits (new features, bug fixes, refactoring, etc.) and evaluates complexity using AI models.
- **GitHub Integration**: Synchronizes data from repositories, commits, issues, and branches on GitHub.
- **Member Analysis**: Builds skill profiles for each member based on contribution history.
- **Risk Assessment**: Analyzes commits and code changes to provide early warnings of potential risks.
- **Task Assignment Suggestions**: Based on skill profiles and task content, the system suggests the most suitable member for execution.
- **Project and Repository Management**: Allows adding, managing, and tracking multiple projects and repositories.

---

## 🛠️ Technologies Used

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React"/>
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite"/>
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

- **Backend**:
  - **Language**: Python 3.9+
  - **Framework**: FastAPI
  - **ORM**: SQLAlchemy with Alembic for migrations.
  - **Environment Management**: Poetry
- **Frontend**:
  - **Framework**: React.js
  - **Build Tool**: Vite
  - **Language**: JavaScript, JSX
  - **Styling**: CSS, with optional libraries like Material-UI or Ant Design.
- **AI & Machine Learning**:
  - **Libraries**: PyTorch/TensorFlow, Scikit-learn, Pandas, NLTK.
  - **Models**:
    - **HAN (Hierarchical Attention Network)**: Used for text classification (commit messages).
    - **MultiFusion Model**: A custom model combining multiple data sources (commit messages, changed files, etc.) for analysis and predictions.
- **Database**: PostgreSQL
- **CI/CD & Deployment**: Docker (planned)

---

## 🏗️ System Architecture

The project is built using a Monorepo architecture, consisting of two main components:

- **`backend/`**: Contains all business logic, API endpoints, AI processing, and database interactions.
- **`frontend/`**: User interface built with React to interact with the backend API.

### **Basic Workflow**

1. Users log in and add a repository from GitHub to the system.
2. The backend synchronizes data (commits, issues, contributors, etc.) from the GitHub API.
3. Data is stored in the PostgreSQL database.
4. AI models process the synchronized data (e.g., commit classification) and save the results.
5. The frontend calls the backend API to display analytical information on the user interface.

---

## 🚀 Installation and Running Guide

### **Requirements**

- Python 3.9+ and Poetry
- Node.js 18+ and npm/yarn
- PostgreSQL Server

### **1. Backend Installation**

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Install dependencies using Poetry
poetry install

# 3. Configure the environment
# Create a .env file and configure necessary environment variables
# (DATABASE_URL, GITHUB_TOKEN, etc.) based on the .env.example file (if available)
cp .env.example .env
# nano .env

# 4. Run database migrations
alembic upgrade head

# 5. Start the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### **2. Frontend Installation**

```bash
# 1. Open another terminal and navigate to the frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Configure the API endpoint
# Open the src/config.js file and ensure the API address points to the backend (http://localhost:8000)

# 4. Start the development server
npm run dev
```

After completing the setup, access `http://localhost:5173` (or the port provided by Vite) in your browser to use the application.

---

## 📂 Directory Structure

```plaintext
.
├── backend/        # Backend source code (FastAPI)
│   ├── ai/         # AI logic, training, prediction
│   ├── api/        # API endpoint definitions (routers)
│   ├── core/       # Common configurations, security, middleware
│   ├── db/         # Database setup, models
│   ├── schemas/    # Pydantic schemas (data validation)
│   ├── services/   # Main business logic
│   └── main.py     # Backend application entry point
├── frontend/       # Frontend source code (React)
│   ├── public/
│   └── src/
│       ├── api/
│       ├── components/
│       ├── features/
│       ├── pages/
│       └── App.jsx
├── docs/           # Project documentation
└── README.md       # This file
```

---

## 📌 Project Information

- **Students**: Vũ Phan Hoài Sang, Lê Trọng Nghĩa
- **Supervisor**: ThS. Đặng Thị Kim Giao
- **University**: HUFLIT (Ho Chi Minh City University of Foreign Languages and Information Technology)
- **Year**: 2025
