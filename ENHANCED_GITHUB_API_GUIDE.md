# Enhanced GitHub API Integration - Lấy Metadata Chi Tiết

## Tổng quan

Hệ thống đã được nâng cấp để lấy metadata chi tiết từ GitHub API, bao gồm:

- `files_changed` - Số lượng file thay đổi
- `additions` - Số dòng code thêm vào
- `deletions` - Số dòng code xóa đi  
- `total_changes` - Tổng số thay đổi (additions + deletions)
- `is_merge` - Có phải merge commit không
- `modified_files` - Danh sách file được thay đổi
- `file_types` - Phân bố loại file (extension)
- `modified_directories` - Phân bố thư mục được thay đổi

## Các Endpoint API Mới

### 1. Lấy Commits với Metadata Chi Tiết

```http
GET /github/{owner}/{repo}/commits/enhanced?branch={branch}&max_commits={n}
```

**Ví dụ:**
```http
GET /github/facebook/react/commits/enhanced?branch=main&max_commits=20
```

**Response:**
```json
{
  "success": true,
  "repository": "facebook/react",
  "branch": "main",
  "commits_count": 20,
  "commits": [
    {
      "sha": "abc123...",
      "message": "feat: add new component",
      "author_name": "Developer",
      "date": "2025-07-02T10:00:00Z",
      "files_changed": 5,
      "additions": 120,
      "deletions": 30,
      "total_changes": 150,
      "is_merge": false,
      "modified_files": [
        "src/components/NewComponent.js",
        "src/index.js",
        "package.json"
      ],
      "file_types": {
        ".js": 2,
        ".json": 1
      },
      "modified_directories": {
        "src/components": 1,
        "src": 1,
        "root": 1
      }
    }
  ]
}
```

### 2. Lấy Chi Tiết File Metadata

```http
GET /github/{owner}/{repo}/commits/{sha}/files
```

**Ví dụ:**
```http
GET /github/facebook/react/commits/abc123def456/files
```

**Response:**
```json
{
  "success": true,
  "repository": "facebook/react",
  "commit_sha": "abc123def456",
  "file_metadata": {
    "files_changed": 5,
    "modified_files": [
      {
        "filename": "src/Component.js",
        "status": "modified",
        "additions": 50,
        "deletions": 10,
        "changes": 60
      }
    ],
    "file_types": {
      ".js": {
        "count": 3,
        "additions": 100,
        "deletions": 20
      }
    },
    "modified_directories": {
      "src": {
        "files": 3,
        "additions": 80,
        "deletions": 15
      }
    },
    "file_categories": {
      "modified": 4,
      "added": 1
    },
    "size_changes": {
      "additions": 120,
      "deletions": 30,
      "total": 150
    }
  }
}
```

### 3. Sync Commits với Enhanced Metadata

```http
POST /github/{owner}/{repo}/sync-enhanced?branch={branch}&max_commits={n}
```

**Ví dụ:**
```http
POST /github/facebook/react/sync-enhanced?branch=main&max_commits=100
```

**Response:**
```json
{
  "success": true,
  "repository": "facebook/react",
  "branch": "main",
  "synced_count": 95,
  "total_fetched": 100,
  "enhanced_metadata": {
    "files_changed": true,
    "file_analysis": true,
    "directory_tracking": true,
    "merge_detection": true,
    "change_statistics": true
  }
}
```

### 4. So Sánh Commits

```http
GET /github/{owner}/{repo}/compare/{base}...{head}
```

**Ví dụ:**
```http
GET /github/facebook/react/compare/abc123...def456
```

## Cách Sử Dụng trong Code

### 1. Lấy Enhanced Commits từ GitHub

```python
from services.github_service import fetch_enhanced_commits_batch

# Lấy 50 commits gần nhất với metadata chi tiết
enhanced_commits = await fetch_enhanced_commits_batch(
    owner="facebook",
    repo="react", 
    branch="main",
    token=github_token,
    max_commits=50
)

for commit in enhanced_commits:
    print(f"Commit {commit['sha'][:8]}:")
    print(f"  Files changed: {commit['files_changed']}")
    print(f"  Total changes: {commit['total_changes']}")
    print(f"  Is merge: {commit['is_merge']}")
    print(f"  File types: {commit['file_types']}")
```

### 2. Lấy Chi Tiết File Metadata

```python
from services.github_service import fetch_commit_files_metadata

file_metadata = await fetch_commit_files_metadata(
    owner="facebook",
    repo="react",
    commit_sha="abc123def456",
    token=github_token
)

print(f"Files changed: {file_metadata['files_changed']}")
print(f"Languages affected: {file_metadata['file_types'].keys()}")
print(f"Directories modified: {file_metadata['modified_directories'].keys()}")
```

### 3. Lưu vào Database với Enhanced Data

```python
from services.commit_service import save_commit

# Commit data từ GitHub API đã có enhanced metadata
commit_data = {
    "sha": "abc123def456",
    "message": "feat: add new feature",
    "author_name": "Developer",
    "author_email": "dev@example.com",
    "repo_id": 1,
    "branch_name": "main",
    # Enhanced metadata
    "files_changed": 5,
    "additions": 120,
    "deletions": 30,
    "total_changes": 150,
    "is_merge": False,
    "modified_files": ["src/App.js", "package.json"],
    "file_types": {".js": 1, ".json": 1},
    "modified_directories": {"src": 1, "root": 1}
}

await save_commit(commit_data)
```

## Tối Ưu Hóa và Rate Limiting

### 1. Batch Processing

```python
# Xử lý commits theo batch để tránh rate limit
batch_size = 10
for i in range(0, len(commit_list), batch_size):
    batch = commit_list[i:i + batch_size]
    
    # Xử lý batch với delay
    enhanced_batch = await fetch_enhanced_commits_batch(
        owner=owner, repo=repo, max_commits=batch_size
    )
    
    # Delay giữa các batch
    await asyncio.sleep(1)
```

### 2. Concurrent Processing với Limits

```python
import asyncio
from asyncio import Semaphore

# Giới hạn concurrent requests
semaphore = Semaphore(5)

async def fetch_with_limit(commit_sha):
    async with semaphore:
        return await fetch_commit_details(commit_sha, owner, repo, token)

# Fetch nhiều commits song song nhưng có giới hạn
tasks = [fetch_with_limit(sha) for sha in commit_shas]
results = await asyncio.gather(*tasks)
```

## Database Schema

### Enhanced Commits Table

```sql
-- Các trường mới đã được thêm vào bảng commits
ALTER TABLE commits ADD COLUMN modified_files JSON;
ALTER TABLE commits ADD COLUMN file_types JSON;
ALTER TABLE commits ADD COLUMN modified_directories JSON;
ALTER TABLE commits ADD COLUMN total_changes INTEGER;
ALTER TABLE commits ADD COLUMN change_type VARCHAR(50);
ALTER TABLE commits ADD COLUMN commit_size VARCHAR(20);
```

### Ví dụ Data Structure

```json
{
  "modified_files": [
    "src/components/Button.js",
    "src/styles/main.css",
    "package.json"
  ],
  "file_types": {
    ".js": 1,
    ".css": 1, 
    ".json": 1
  },
  "modified_directories": {
    "src/components": 1,
    "src/styles": 1,
    "root": 1
  }
}
```

## Error Handling

### 1. GitHub API Rate Limits

```python
try:
    commits = await fetch_enhanced_commits_batch(owner, repo)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        # Rate limit exceeded
        wait_time = int(e.response.headers.get("Retry-After", 60))
        await asyncio.sleep(wait_time)
        # Retry
```

### 2. Missing Data Handling

```python
# Fallback khi không lấy được enhanced data
enhanced_metadata = commit_data.get("enhanced_metadata", {})
files_changed = enhanced_metadata.get("files_changed", 0)
modified_files = enhanced_metadata.get("modified_files", [])

# Sử dụng analyzer nếu không có data từ GitHub
if not modified_files and commit_data.get("files"):
    analyzer_metadata = CommitAnalyzer.extract_commit_metadata(commit_data)
    modified_files = analyzer_metadata.get("modified_files", [])
```

## Testing

Chạy test script để kiểm tra integration:

```bash
cd backend
python scripts/test_enhanced_github_api.py
```

## Performance Monitoring

### 1. API Usage Tracking

```python
# Monitor API calls and response times
import time

start_time = time.time()
enhanced_commits = await fetch_enhanced_commits_batch(owner, repo)
end_time = time.time()

print(f"Fetched {len(enhanced_commits)} commits in {end_time - start_time:.2f}s")
```

### 2. Rate Limit Monitoring

```python
from services.github_service import get_rate_limit_info

rate_limit = await get_rate_limit_info(token)
remaining = rate_limit.get("resources", {}).get("core", {}).get("remaining", 0)
print(f"API calls remaining: {remaining}")
```

## Lợi Ích

1. **Dữ liệu Phong Phú**: Metadata chi tiết về từng commit
2. **Phân Tích Sâu**: Hiểu rõ impact của các thay đổi code
3. **Tracking Files**: Theo dõi files và directories được thay đổi
4. **Performance Insights**: Phân tích patterns trong development
5. **Quality Metrics**: Đánh giá chất lượng commits và code changes

## Kết Luận

Với enhanced GitHub API integration, bạn có thể:

- Lấy metadata chi tiết từ GitHub API
- Phân tích sâu về file changes và directory impact  
- Tracking comprehensive commit statistics
- Identify merge commits và branching patterns
- Monitor development patterns và code quality metrics

Hệ thống này cung cấp foundation mạnh mẽ cho advanced analytics và insights về development process.
