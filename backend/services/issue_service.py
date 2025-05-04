from db.database import database
from db.models import issues

# Lưu một issue duy nhất
async def save_issue(issue_data):
    query = issues.insert().values(
        github_id=issue_data["github_id"],
        title=issue_data["title"],
        body=issue_data["body"],
        state=issue_data["state"],
        created_at=issue_data["created_at"],
        updated_at=issue_data["updated_at"],
        repo_id=issue_data["repo_id"],
    )
    await database.execute(query)

# Lưu danh sách nhiều issue
async def save_issues(issue_list):
    for issue in issue_list:
        await save_issue(issue)
