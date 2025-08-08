# Báo cáo Cơ sở Dữ liệu Hệ thống

## 1. Tổng quan

Hệ thống sử dụng **PostgreSQL** làm cơ sở dữ liệu chính với **SQLAlchemy** làm ORM và **Alembic** để quản lý migration. Thiết kế database hướng đến việc lưu trữ và phân tích dữ liệu từ GitHub repositories một cách hiệu quả.

### Thông tin kỹ thuật:

- **Database Engine**: PostgreSQL (với fallback SQLite cho development)
- **ORM**: SQLAlchemy (Core Table approach)
- **Migration Tool**: Alembic
- **Connection Pool**: SQLAlchemy SessionLocal
- **Async Support**: Databases library với asyncpg driver

**Lưu ý**: Hệ thống có 10 bảng chính được sử dụng tích cực, không bao gồm bảng `issues` vì không được implement trong application chính.

## 2. Kiến trúc Cơ sở Dữ liệu

### 2.1. Cấu trúc Kết nối

```python
# Database Configuration
DATABASE_URL = "postgresql+asyncpg://user:password@host:port/database"
# Fallback for sync operations: postgresql+psycopg2://...

# Dual Engine Setup
sync_engine = create_engine(DATABASE_URL.replace("asyncpg", "psycopg2"))
database = Database(DATABASE_URL)  # Async operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
```

### 2.2. Metadata Management

Sử dụng **Shared Metadata** pattern để đảm bảo consistency:

```python
# backend/db/metadata.py
metadata = MetaData()  # Single source of truth cho tất cả tables
```

## 3. Schema Database

Hệ thống sử dụng **10 bảng chính** được chia thành các nhóm chức năng:

### 3.1. Bảng Chính (Core Tables)

#### **users** - Quản lý người dùng

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    github_id INTEGER,
    github_username VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    display_name VARCHAR(255),
    full_name VARCHAR(255),
    avatar_url VARCHAR(500),
    bio TEXT,
    location VARCHAR(255),
    company VARCHAR(255),
    blog VARCHAR(500),
    twitter_username VARCHAR(255),
    github_profile_url VARCHAR(500),
    repos_url VARCHAR(500),
    is_active BOOLEAN,
    is_verified BOOLEAN,
    github_created_at TIMESTAMP,
    last_synced TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Mục đích**: Lưu trữ thông tin người dùng đã đăng nhập vào hệ thống và liên kết với GitHub profile.

#### **repositories** - Quản lý repositories

```sql
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    github_id INTEGER NOT NULL,
    owner VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    full_name VARCHAR(500),
    description TEXT,
    stars INTEGER,
    forks INTEGER,
    language VARCHAR(100),
    open_issues INTEGER,
    url VARCHAR(500),
    clone_url VARCHAR(500),
    is_private BOOLEAN,
    is_fork BOOLEAN,
    default_branch VARCHAR(100),
    last_synced TIMESTAMP,
    sync_status VARCHAR(20),
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Mục đích**: Lưu trữ metadata của các GitHub repositories được phân tích.

#### **commits** - Dữ liệu commit (Bảng quan trọng nhất)

```sql
CREATE TABLE commits (
    id SERIAL PRIMARY KEY,
    sha VARCHAR(40) NOT NULL,
    message TEXT NOT NULL,
    author_user_id INTEGER REFERENCES users(id),
    author_name VARCHAR(255) NOT NULL,
    author_email VARCHAR(255) NOT NULL,
    committer_user_id INTEGER REFERENCES users(id),
    committer_name VARCHAR(255),
    committer_email VARCHAR(255),
    repo_id INTEGER REFERENCES repositories(id) NOT NULL,
    branch_id INTEGER REFERENCES branches(id),
    branch_name VARCHAR(255),
    author_role_at_commit VARCHAR(20),
    author_permissions_at_commit VARCHAR(100),
    date TIMESTAMP NOT NULL,
    committer_date TIMESTAMP,
    insertions INTEGER,
    deletions INTEGER,
    files_changed INTEGER,
    parent_sha VARCHAR(40),
    is_merge BOOLEAN,
    merge_from_branch VARCHAR(255),

    -- Enhanced fields for AI analysis
    modified_files JSON,  -- List of modified file paths
    file_types JSON,      -- Dictionary of file extensions and counts
    modified_directories JSON,  -- Dictionary of directories and change counts
    total_changes INTEGER,      -- Total additions + deletions
    change_type VARCHAR(50),    -- feature, bugfix, refactor, etc.
    commit_size VARCHAR(20),    -- small, medium, large

    created_at TIMESTAMP DEFAULT NOW(),
    last_synced TIMESTAMP DEFAULT NOW()
);
```

**Đặc biệt**: Bảng này có **enhanced fields** được thiết kế đặc biệt cho AI analysis với JSON columns để lưu structured data.

### 3.2. Bảng Collaborator System

#### **collaborators** - Canonical source cho GitHub users

```sql
CREATE TABLE collaborators (
    id SERIAL PRIMARY KEY,
    github_user_id INTEGER UNIQUE NOT NULL,
    github_username VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    email VARCHAR(255),
    avatar_url VARCHAR(500),
    bio TEXT,
    company VARCHAR(255),
    location VARCHAR(255),
    blog VARCHAR(500),
    is_site_admin BOOLEAN DEFAULT FALSE,
    node_id VARCHAR(255),
    gravatar_id VARCHAR(255),
    type VARCHAR(50) DEFAULT 'User',
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_collaborators_github_user_id ON collaborators(github_user_id);
CREATE INDEX idx_collaborators_github_username ON collaborators(github_username);
```

#### **repository_collaborators** - Mapping collaborators to repositories

```sql
CREATE TABLE repository_collaborators (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER REFERENCES repositories(id) NOT NULL,
    collaborator_id INTEGER REFERENCES collaborators(id) NOT NULL,
    role VARCHAR(50) NOT NULL,  -- admin, maintain, push, triage, pull
    permissions VARCHAR(100),
    is_owner BOOLEAN DEFAULT FALSE,
    joined_at TIMESTAMP,
    invited_by VARCHAR(255),
    invitation_status VARCHAR(20),
    commits_count INTEGER DEFAULT 0,
    issues_count INTEGER DEFAULT 0,
    prs_count INTEGER DEFAULT 0,
    last_activity TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_synced TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_repo_collaborators_repo ON repository_collaborators(repository_id);
CREATE INDEX idx_repo_collaborators_collaborator ON repository_collaborators(collaborator_id);
CREATE UNIQUE INDEX idx_repo_collaborators_unique ON repository_collaborators(repository_id, collaborator_id);
```

### 3.3. Bảng Quản lý Dự án

#### **branches** - Quản lý branches

```sql
CREATE TABLE branches (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    repo_id INTEGER REFERENCES repositories(id) NOT NULL,
    creator_user_id INTEGER REFERENCES users(id),
    creator_name VARCHAR(255),
    last_committer_user_id INTEGER REFERENCES users(id),
    last_committer_name VARCHAR(255),
    sha VARCHAR(40),
    is_default BOOLEAN,
    is_protected BOOLEAN,
    created_at TIMESTAMP,
    last_commit_date TIMESTAMP,
    last_synced TIMESTAMP DEFAULT NOW(),
    commits_count INTEGER,
    contributors_count INTEGER
);
```

#### **project_tasks** - Task management system

```sql
CREATE TABLE project_tasks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    assignee_user_id INTEGER REFERENCES users(id),
    assignee_github_username VARCHAR(100),
    status VARCHAR(11) NOT NULL,  -- TODO, IN_PROGRESS, DONE, CANCELLED
    priority VARCHAR(6) NOT NULL, -- LOW, MEDIUM, HIGH, URGENT
    due_date VARCHAR(10),
    repository_id INTEGER REFERENCES repositories(id),
    repo_owner VARCHAR(100),
    repo_name VARCHAR(100),
    is_completed BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by_user_id INTEGER REFERENCES users(id),
    created_by VARCHAR(100)
);
```

### 3.4. Bảng GitHub Integration

#### **pull_requests** - GitHub Pull Requests

```sql
CREATE TABLE pull_requests (
    id SERIAL PRIMARY KEY,
    github_id INTEGER,
    title VARCHAR(255) NOT NULL,
    description VARCHAR(255),
    state VARCHAR(50),
    repo_id INTEGER REFERENCES repositories(id) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 3.5. Bảng Utility

#### **assignments** - Simple assignment tracking

```sql
CREATE TABLE assignments (
    id SERIAL PRIMARY KEY,
    task_name VARCHAR(255) NOT NULL,
    description VARCHAR(255),
    is_completed BOOLEAN,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### **user_repositories** - User-Repository relationships

```sql
CREATE TABLE user_repositories (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    repository_id INTEGER REFERENCES repositories(id) NOT NULL,
    role VARCHAR(12) NOT NULL,
    permissions VARCHAR(5) NOT NULL,
    is_primary_owner BOOLEAN,
    joined_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## 4. Relationship Diagram

```mermaid
erDiagram
    users ||--o{ repositories : owns
    users ||--o{ commits : authors
    users ||--o{ commits : commits
    users ||--o{ project_tasks : assigned_to
    users ||--o{ assignments : has
    users ||--o{ user_repositories : belongs_to
    users ||--o{ collaborators : links_to

    repositories ||--o{ commits : contains
    repositories ||--o{ branches : has
    repositories ||--o{ pull_requests : has
    repositories ||--o{ project_tasks : relates_to
    repositories ||--o{ repository_collaborators : has
    repositories ||--o{ user_repositories : linked_by

    branches ||--o{ commits : contains

    collaborators ||--o{ repository_collaborators : participates_in

    repositories }o--o{ collaborators : repository_collaborators
```

## 5. Đặc điểm Kỹ thuật

### 5.1. JSON Fields cho AI Analysis

Bảng `commits` sử dụng JSON fields để lưu trữ structured data:

```python
# Trong SQLAlchemy model
def JSONType():
    """Return appropriate JSON type based on database dialect"""
    return JSON().with_variant(Text(), "sqlite")

# Usage
Column('modified_files', JSONType(), nullable=True,
       comment='List of modified file paths')
Column('file_types', JSONType(), nullable=True,
       comment='Dictionary of file extensions and their counts')
```

### 5.2. Cross-Database Compatibility

System được thiết kế để work với cả PostgreSQL và SQLite:

```python
# Migration example
if dialect_name == 'postgresql':
    json_type = postgresql.JSON()
else:  # SQLite fallback
    json_type = sa.TEXT()
```

### 5.3. Index Strategy

Các index được thiết kế cho performance:

- `github_user_id` trên collaborators (unique)
- `github_username` trên collaborators
- `repository_id, collaborator_id` composite index
- Repository và user foreign keys

### 5.4. Timestamp Management

Consistent timestamp handling:

- `created_at` - record creation time
- `updated_at` - last modification time
- `last_synced` - last GitHub sync time
- Auto-update với `server_default=func.now()`

## 6. Migration Management

### 6.1. Alembic Configuration

```python
# migrations/env.py
target_metadata = metadata  # Từ shared metadata
database_url = os.getenv("DATABASE_URL").replace("asyncpg", "psycopg2")
```

### 6.2. Migration History

1. **Initial Migration**: `a989fa2a380c_initial_migration_with_all_models.py`

   - Tạo tất cả tables cơ bản
   - Setup relationships và constraints

2. **Enhanced Commit Fields**: `add_enhanced_commit_fields.py`
   - Thêm JSON fields cho AI analysis
   - Cross-database compatibility

### 6.3. Migration Commands

```bash
# Generate new migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Check current version
alembic current
```

## 7. Data Flow và Use Cases

### 7.1. GitHub Data Sync Flow

1. **Repository Sync**: GitHub API → `repositories` table
2. **User Sync**: GitHub users → `collaborators` → link to `users`
3. **Commit Sync**: GitHub commits → `commits` với enhanced analysis
4. **Relationship Sync**: Collaborators → `repository_collaborators`
5. **Pull Request Sync**: GitHub PRs → `pull_requests` table (khi cần)

### 7.2. AI Analysis Pipeline

1. **Input**: Raw commit data từ `commits` table
2. **Processing**: AI models analyze commit messages + code metrics
3. **Output**: Enhanced fields (`change_type`, `commit_size`, JSON metadata)
4. **Storage**: Back to `commits` table enhanced fields

### 7.3. User Management Flow

1. **Authentication**: GitHub OAuth → `users` table
2. **Repository Access**: User permissions → `user_repositories`
3. **Project Management**: Tasks → `project_tasks` + `assignments`

## 8. Performance Considerations

### 8.1. Query Optimization

- Foreign key indexes cho joins
- Composite indexes cho complex queries
- JSON field indexing (PostgreSQL specific)

### 8.2. Connection Management

```python
# Async + Sync dual approach
database = Database(DATABASE_URL)  # For async operations
SessionLocal = sessionmaker(bind=sync_engine)  # For ORM operations
```

### 8.3. Data Archiving Strategy

- `last_synced` fields để track data freshness
- Incremental sync thay vì full refresh
- Soft delete pattern cho historical data

## 9. Security và Compliance

### 9.1. Data Privacy

- GitHub tokens stored securely (not in database)
- User email protection
- GDPR compliance ready (user data deletion)

### 9.2. Access Control

- Repository-based permissions
- User role management trong `repository_collaborators`
- OAuth integration cho authentication

## 10. Monitoring và Maintenance

### 10.1. Health Checks

```python
# Database connection check
try:
    with sync_engine.connect() as connection:
        print("Kết nối cơ sở dữ liệu thành công!")
except Exception as e:
    print(f"Lỗi kết nối cơ sở dữ liệu: {e}")
```

### 10.2. Backup Strategy

- PostgreSQL automated backups
- Migration rollback capability
- Data export utilities

### 10.3. Performance Monitoring

- Connection pool monitoring
- Query performance tracking
- Index usage analysis

## 11. Kết luận

Cơ sở dữ liệu được thiết kế với các đặc điểm chính:

1. **Scalable**: Hỗ trợ large repositories với many commits
2. **AI-Ready**: JSON fields cho machine learning analysis
3. **GitHub-Native**: Direct mapping với GitHub API responses
4. **Cross-Platform**: PostgreSQL primary, SQLite fallback
5. **Migration-Friendly**: Alembic-based schema evolution
6. **Performance-Optimized**: Strategic indexing và relationship design
7. **Focused Design**: 10 bảng core tập trung vào commit analysis và user management

Database schema này cung cấp foundation mạnh mẽ cho GitHub analytics platform với khả năng mở rộng cho future features và AI enhancements. Việc loại bỏ bảng `issues` giúp giữ schema đơn giản và tập trung vào core functionality.
