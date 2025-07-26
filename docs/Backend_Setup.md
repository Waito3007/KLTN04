# Backend Setup Guide

This guide provides instructions on how to set up and run the backend application. The backend is built with Python, FastAPI, and uses PostgreSQL as its database.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.9+**: [Download Python](https://www.python.org/downloads/)
*   **Poetry**: A dependency management and packaging tool for Python. [Poetry Installation Guide](https://python-poetry.org/docs/#installation)
*   **PostgreSQL**: A powerful, open-source object-relational database system. [PostgreSQL Downloads](https://www.postgresql.org/download/)
*   **Git**: For cloning the repository. [Git Downloads](https://git-scm.com/downloads)

## 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone <repository_url>
cd KLTN04/backend
```

## 2. Set up Python Environment and Install Dependencies

Navigate to the `backend` directory and install dependencies using Poetry:

```bash
cd E:/Dự Án Của Nghĩa/KLTN04/backend
poetry install
poetry shell
```

This will create a virtual environment and install all necessary packages defined in `pyproject.toml` and `poetry.lock`.

## 3. Database Configuration

### 3.1. Create PostgreSQL Database

Create a new PostgreSQL database for the project. For example, you can name it `kltn04_db`.

```sql
CREATE DATABASE kltn04_db;
```

### 3.2. Environment Variables

Create a `.env` file in the `backend` directory (`E:/Dự Án Của Nghĩa/KLTN04/backend/.env`) and add your database connection details and other configurations. Replace the placeholders with your actual values.

```dotenv
DATABASE_URL="postgresql://user:password@host:port/kltn04_db"
# Example: DATABASE_URL="postgresql://postgres:mysecretpassword@localhost:5432/kltn04_db"

# GitHub OAuth App Credentials (if applicable)
GITHUB_CLIENT_ID="your_github_client_id"
GITHUB_CLIENT_SECRET="your_github_client_secret"

# Secret key for JWT token encryption
SECRET_KEY="your_super_secret_key_for_jwt"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 3.3. Run Database Migrations

Apply the database migrations using Alembic to create the necessary tables:

```bash
alembic upgrade head
```

## 4. Run the Backend Application

Once the database is set up, you can start the FastAPI application using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*   `main:app`: Refers to the `app` object in `main.py`.
*   `--host 0.0.0.0`: Makes the server accessible from other devices on the network (use `127.0.0.1` or `localhost` for local access only).
*   `--port 8000`: Runs the server on port 8000.
*   `--reload`: Enables auto-reloading on code changes (useful for development).

The backend API will now be accessible at `http://localhost:8000`.

## 5. Access API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, you can access it at:

*   **Swagger UI**: `http://localhost:8000/docs`
*   **ReDoc**: `http://localhost:8000/redoc`
