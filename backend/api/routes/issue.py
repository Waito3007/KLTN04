# backend/api/routes/issue.py
from fastapi import APIRouter, Request, HTTPException
import httpx
from services.issue_service import save_issue
from services.repo_service import get_repo_id_by_owner_and_name

issue_router = APIRouter()

# Lưu issues vào database
@issue_router.post("/github/{owner}/{repo}/save-issues")
async def save_issues(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    # Lấy danh sách issue từ GitHub API
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        issues = resp.json()

    # Lưu issue vào database
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    saved_count = 0
    for issue in issues:
        try:
            issue_data = {
                "title": issue["title"],
                "body": issue["body"],
                "state": issue["state"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "repo_id": repo_id,
            }
            await save_issue(issue_data)
            saved_count += 1
        except Exception as e:
            print(f"Lỗi khi lưu issue {issue['title']}: {e}")
            continue

    return {"message": f"Đã lưu {saved_count}/{len(issues)} issues!"}
