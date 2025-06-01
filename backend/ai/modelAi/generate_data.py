import json
from db.database import database
from db.models.commits import commits  # Đảm bảo bạn đã import đúng model commits
from datetime import datetime
import re

# Hàm để chuyển dữ liệu commit thành định dạng spaCy
async def convert_to_spacy_format():
    # Kết nối cơ sở dữ liệu trước khi truy vấn
    await database.connect()

    # Truy vấn tất cả dữ liệu commits từ database
    query = commits.select()  # Thay đổi nếu bạn sử dụng ORM khác như SQLAlchemy
    rows = await database.fetch_all(query)

    # Danh sách để lưu dữ liệu theo định dạng spaCy
    data = []

    for row in rows:
        # Lấy dữ liệu từ commit
        sha = row["sha"]
        message = row["message"]  # Lấy trường 'message' từ commit
        author_name = row["author_name"]
        author_email = row["author_email"]
        date = row["date"]  # Trường ngày giờ commit
        insertions = row["insertions"]
        deletions = row["deletions"]
        files_changed = row["files_changed"]
        repo_id = row["repo_id"]

        # Nếu date là kiểu datetime, chuyển nó thành chuỗi
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d %H:%M:%S')  # Chuyển đổi theo định dạng mong muốn
        
        # Kiểm tra commit có chứa trường message hay không
        if message:
            # Phân loại label từ message
            label = get_label_from_message(message)

            # Thêm vào danh sách data với định dạng spaCy
            data.append({
                "text": message,  # Nội dung commit
                "meta": {
                    "sha": sha,  # SHA của commit
                    "author_name": author_name,  # Tên tác giả
                    "author_email": author_email,  # Email tác giả
                    "date": date,  # Ngày commit (chuỗi)
                    "insertions": insertions,  # Số dòng thêm
                    "deletions": deletions,  # Số dòng xóa
                    "files_changed": files_changed,  # Số file thay đổi
                    "repo_id": repo_id  # ID của repository
                },
                "cats": label  # Thẻ phân loại của commit
            })

    # Ghi dữ liệu ra file JSON cho spaCy
    with open("ai/train_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Dữ liệu đã được chuyển thành công!")

    # Ngắt kết nối cơ sở dữ liệu sau khi hoàn thành
    await database.disconnect()

# Hàm thu thập commit messages và lưu theo định dạng unified JSON
async def export_commit_messages_unified():
    await database.connect()
    query = commits.select()
    rows = await database.fetch_all(query)
    data = []
    for row in rows:
        sha = row["sha"]
        message = row["message"]
        author_name = row["author_name"]
        author_email = row["author_email"]
        date = row["date"]
        insertions = row["insertions"]
        deletions = row["deletions"]
        files_changed = row["files_changed"]
        repo_id = row["repo_id"]
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d %H:%M:%S')
        if message:
            data.append({
                "id": sha,
                "data_type": "commit_message",
                "raw_text": message,
                "source_info": {
                    "repo_id": repo_id,
                    "sha": sha,
                    "author_name": author_name,
                    "author_email": author_email,
                    "date": date,
                    "insertions": insertions,
                    "deletions": deletions,
                    "files_changed": files_changed
                },
                "labels": {
                    "purpose": None,
                    "suspicious": None,
                    "tech_tag": None,
                    "sentiment": None
                }
            })
    with open("ai/collected_data/commit_messages_from_owner.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Đã xuất commit messages sang collected_data/commit_messages_raw.json!")
    await database.disconnect()

# Hàm phân loại label từ message (cần phải viết logic phân loại của bạn)
def get_label_from_message(message):
    # Các từ khóa phổ biến trong commit
    categories = {
        "feat": {"feat": 1.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "thêm": {"feat": 1.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "fix": {"feat": 0.0, "fix": 1.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "sửa": {"feat": 0.0, "fix": 1.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "docs": {"feat": 0.0, "fix": 0.0, "docs": 1.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "style": {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 1.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "refactor": {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 1.0, "chore": 0.0, "test": 0.0},
        "chore": {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 1.0, "test": 0.0},
        "test": {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 1.0},
    }

    # Kiểm tra nếu commit chứa một trong các từ khóa trên
    for keyword, label in categories.items():
        if keyword in message.lower():
            return label
    
    # Nếu không có từ khóa nào trong message, phân loại là 'uncategorized'
    return {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0, "uncategorized": 1.0}

def classify_commit_purpose(message: str) -> str:
    message = message.lower()
    # Bug Fix
    if re.search(r"\b(fix|bug|error|sửa lỗi|vá lỗi|issue #|problem|resolve|patch|hotfix|defect|repair|correct|debug)\b", message):
        return 'Bug Fix'
    # Feature Implementation
    if re.search(r"\b(feat|add|implement|thêm|phát triển|tính năng mới|feature|introduce|support|create|build)\b", message):
        return 'Feature Implementation'
    # Refactoring
    if re.search(r"\b(refactor|restructure|tái cấu trúc|optimi[sz]e|clean up|cleanup|improve structure|refactoring)\b", message):
        return 'Refactoring'
    # Documentation Update
    if re.search(r"\b(docs|update readme|document|tài liệu|hướng dẫn|readme|docstring|documentation|manual|guide)\b", message):
        return 'Documentation Update'
    # Test Update
    if re.search(r"\b(test|unit test|integration test|add test|update test|kiểm thử|bổ sung test|testcase|test case|testing)\b", message):
        return 'Test Update'
    # Security Patch
    if re.search(r"\b(security|bảo mật|vulnerability|cve-|patch security|fix security|secure|xss|csrf|injection|auth bypass|exploit)\b", message):
        return 'Security Patch'
    # Code Style/Formatting
    if re.search(r"\b(style|format|formatting|code style|reformat|lint|prettier|black|flake8|isort|format code|format lại|chuẩn hóa mã)\b", message):
        return 'Code Style/Formatting'
    # Build/CI/CD Script Update
    if re.search(r"\b(build|ci|cd|pipeline|deploy|release|workflow|github actions|jenkins|travis|circleci|update script|build script|ci/cd|docker|compose|build system|deployment)\b", message):
        return 'Build/CI/CD Script Update'
    # Default fallback
    return 'Other'

def is_suspicious_commit(message: str) -> int:
    if not message or len(message.strip()) < 5:
        return 1  # Quá ngắn hoặc không có nội dung
    if len(message) > 200:
        return 1  # Quá dài bất thường
    suspicious_keywords = [
        'hack', 'backdoor', 'temp fix', 'quick fix', 'todo: fix', 'xxx', 'hack', 'workaround',
        'bypass', 'disable security', 'hardcode', 'debug', 'remove check', 'skip test', 'patch quick', 'urgent fix'
    ]
    msg_lower = message.lower()
    for kw in suspicious_keywords:
        if kw in msg_lower:
            return 1
    # Toàn bộ viết hoa hoặc chứa nhiều ký tự đặc biệt
    if message.isupper() or sum(1 for c in message if not c.isalnum() and c not in ' .,:;') > len(message) * 0.3:
        return 1
    return 0

def extract_tech_tags(text: str) -> list:
    tech_vocab = [
        'python', 'fastapi', 'react', 'javascript', 'typescript', 'docker', 'sqlalchemy', 'pytorch', 'spacy',
        'css', 'html', 'postgresql', 'mysql', 'mongodb', 'redis', 'vue', 'angular', 'flask', 'django',
        'node', 'express', 'graphql', 'rest', 'api', 'gitlab', 'github', 'ci', 'cd', 'kubernetes', 'helm',
        'pytest', 'unittest', 'junit', 'cicd', 'github actions', 'travis', 'jenkins', 'circleci', 'webpack',
        'babel', 'vite', 'npm', 'yarn', 'pip', 'poetry', 'black', 'flake8', 'isort', 'prettier', 'eslint',
        'jwt', 'oauth', 'sso', 'celery', 'rabbitmq', 'kafka', 'grpc', 'protobuf', 'swagger', 'openapi',
        'sentry', 'prometheus', 'grafana', 'nginx', 'apache', 'linux', 'ubuntu', 'windows', 'macos',
        'aws', 'azure', 'gcp', 'firebase', 'heroku', 'netlify', 'vercel', 'tailwind', 'bootstrap', 'material ui'
    ]
    found = set()
    text_lower = text.lower()
    for tech in tech_vocab:
        if tech in text_lower:
            found.add(tech)
    return list(found)

def classify_sentiment(message: str) -> str:
    message = message.lower()
    positive_keywords = [
        'cải thiện', 'tốt', 'thành công', 'hoàn thành', 'ổn định', 'tối ưu', 'đẹp', 'gọn', 'sạch', 'great', 'awesome', 'improved', 'cleaned up', 'ok', 'hoàn tất', 'đã xong', 'resolved', 'fixed', 'passed', 'success', 'hoan thanh', 'tot', 'cam on', 'thanks', 'thank you', 'well done', 'hoàn thiện', 'đúng', 'đúng chức năng', 'đúng yêu cầu'
    ]
    negative_keywords = [
        'lỗi', 'fail', 'broke', 'revert', 'issue', 'bug', 'không chạy', 'không hoạt động', 'sai', 'chưa xong', 'chưa hoàn thành', 'chưa đúng', 'problem', 'error', 'crash', 'exception', 'bad', 'xóa bỏ', 'rollback', 'undo', 'fixme', 'todo', 'tạm thời', 'workaround', 'hack', 'tạm vá', 'không ổn', 'không tốt', 'chưa ổn', 'chưa tốt', 'chưa hoàn thiện', 'chưa đúng chức năng', 'chưa đúng yêu cầu'
    ]
    for word in positive_keywords:
        if word in message:
            return 'positive'
    for word in negative_keywords:
        if word in message:
            return 'negative'
    return 'neutral'

# Chạy hàm convert_to_spacy_format
if __name__ == "__main__":
    import asyncio
    #asyncio.run(convert_to_spacy_format())
    asyncio.run(export_commit_messages_unified())
