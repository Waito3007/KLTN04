from fastapi import APIRouter, HTTPException, UploadFile, File, Header, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
from pathlib import Path
import tempfile
import httpx
import asyncio
from datetime import datetime
from api.routes.commit import fetch_raw_github_content
from services.model_loader import predict_commit as _model_predict_commit

# Wrapper chuẩn hoá kết quả dự đoán về kiểu int 0/1
def predict_commit(message: str) -> int:
    """Dự đoán commit critical (0|1) dựa trên model_loader.predict_commit.

    model_loader.predict_commit trả về dict {prediction, confidence, error}.
    Hàm này chuẩn hoá để các endpoint dùng an toàn.
    """
    try:
        result = _model_predict_commit(message)
        # Trường hợp model_loader trả về dict chuẩn
        if isinstance(result, dict):
            pred = result.get("prediction", 0)
            return 1 if int(pred) == 1 else 0
        # Trường hợp khác (phòng thủ)
        return 1 if int(result) == 1 else 0
    except Exception:
        return 0

router = APIRouter(prefix="/api/commits", tags=["Commit Analysis"])

@router.get("/analyze-github/{owner}/{repo}")
async def analyze_github_commits(
    owner: str,
    repo: str,
    authorization: str = Header(..., alias="Authorization"),
    per_page: int = 30,
    since: Optional[str] = None,
    until: Optional[str] = None,
    include_diff: bool = False  # New parameter
):
    """
    Phân tích commit từ repository GitHub
    
    Args:
        owner: Tên chủ repo
        repo: Tên repository
        authorization: Token GitHub (Format: Bearer <token>)
        per_page: Số commit tối đa cần phân tích (1-100)
        since: Lọc commit từ ngày (YYYY-MM-DDTHH:MM:SSZ)
        until: Lọc commit đến ngày (YYYY-MM-DDTHH:MM:SSZ)
        include_diff: Có bao gồm nội dung diff đầy đủ của mỗi commit hay không (mặc định: False)
    
    Returns:
        {
            "repo": f"{owner}/{repo}",
            "total": int,
            "critical": int,
            "critical_percentage": float,
            "details": List[dict],
            "analysis_date": str
        }
    """
    try:
        # Validate input
        if per_page < 1 or per_page > 100:
            raise HTTPException(
                status_code=400,
                detail="per_page must be between 1 and 100"
            )

        # Extract token from Authorization header
        token = authorization.replace("Bearer ", "")
        if not token:
            raise HTTPException(status_code=401, detail="Missing or invalid GitHub token")

        # Configure GitHub API request
        headers = {
            "Authorization": authorization,
            "Accept": "application/vnd.github.v3+json"
        }
        params = {
            "per_page": per_page,
            "since": since,
            "until": until
        }
        
        # Fetch commits from GitHub
        async with httpx.AsyncClient() as client:
            # Get first page to check repo accessibility
            initial_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            response = await client.get(initial_url, headers=headers, params={**params, "per_page": 1})
            
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail="Repository not found or access denied"
                )
            response.raise_for_status()

            # Get all requested commits
            full_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            response = await client.get(full_url, headers=headers, params=params)
            response.raise_for_status()
            commits_data = response.json()

        # Prepare analysis data
        commits_for_analysis = []
        for commit_data in commits_data:
            commit_sha = commit_data.get("sha")
            commit_message = commit_data.get("commit", {}).get("message")
            commit_date = commit_data.get("commit", {}).get("committer", {}).get("date")

            if commit_sha and commit_message:
                commit_info = {
                    "id": commit_sha,
                    "message": commit_message,
                    "date": commit_date
                }
                
                if include_diff:
                    try:
                        # Fetch full diff content for the commit
                        diff_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
                        full_commit_details = await fetch_raw_github_content(diff_url, token)
                        commit_info["diff_content"] = full_commit_details
                    except Exception as e:
                        # Log error but don't fail the whole request
                        print(f"Warning: Could not fetch diff for commit {commit_sha}: {e}")
                        commit_info["diff_content"] = "Error fetching diff"
                
                commits_for_analysis.append(commit_info)

        # Analyze commits
        results = {
            "repo": f"{owner}/{repo}",
            "total": len(commits_for_analysis),
            "critical": 0,
            "critical_percentage": 0.0,
            "details": [],
            "analysis_date": datetime.utcnow().isoformat()
        }

        for commit in commits_for_analysis:
            is_critical = predict_commit(commit['message'])
            if is_critical:
                results["critical"] += 1
            
            detail_entry = {
                "id": commit["id"],
                "is_critical": is_critical,
                "message_preview": commit['message'][:100] + "..." if len(commit['message']) > 100 else commit['message'],
                "date": commit["date"]
            }
            if include_diff:
                detail_entry["diff_content"] = commit.get("diff_content", "")
            
            results["details"].append(detail_entry)

        # Calculate percentage
        if results["total"] > 0:
            results["critical_percentage"] = round(
                (results["critical"] / results["total"]) * 100, 2
            )

        return results

    except httpx.HTTPStatusError as e:
        error_detail = "GitHub API error"
        if e.response.status_code == 403:
            error_detail = "API rate limit exceeded" if "rate limit" in str(e.response.content) else "Forbidden"
        elif e.response.status_code == 401:
            error_detail = "Invalid GitHub token"
        
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"{error_detail}: {e.response.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing GitHub commits: {str(e)}"
        )
@router.get("/analyze-text")
async def analyze_commit_text(message: str):
    """
    Phân tích một commit message dạng text
    
    Args:
        message: Nội dung commit message (query parameter)
    
    Returns:
        {"is_critical": 0|1, "message": string}
    """
    try:
        is_critical = predict_commit(message)
        return {
            "is_critical": is_critical,
            "message": "Phân tích thành công",
            "input_sample": message[:100] + "..." if len(message) > 100 else message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")

@router.post("/analyze-text-post")
async def analyze_commit_text_post(request_data: dict):
    """
    Phân tích một commit message dạng text (POST method)
    
    Args:
        request_data: {"message": "commit message text"}
    
    Returns:
        {"is_critical": 0|1, "message": string}
    """
    try:
        message = request_data.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="Missing 'message' field")
            
        is_critical = predict_commit(message)
        return {
            "is_critical": is_critical,
            "message": "Phân tích thành công",
            "input_sample": message[:100] + "..." if len(message) > 100 else message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")

@router.post("/analyze-json")
async def analyze_commits_json(commits: List[dict]):
    """
    Phân tích nhiều commit từ JSON
    
    Args:
        commits: List[{"id": string, "message": string}]
    
    Returns:
        {"total": int, "critical": int, "details": List[dict]}
    """
    try:
        results = {
            "total": len(commits),
            "critical": 0,
            "details": []
        }
        
        for commit in commits:
            if not isinstance(commit, dict) or 'message' not in commit:
                continue
                
            is_critical = predict_commit(commit['message'])
            if is_critical:
                results["critical"] += 1
                
            results["details"].append({
                "id": commit.get("id", ""),
                "is_critical": is_critical,
                "message_preview": commit['message'][:100] + "..." if len(commit['message']) > 100 else commit['message']
            })
            
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích hàng loạt: {str(e)}")

@router.post("/analyze-csv", response_model=dict)
async def analyze_commits_csv(file: UploadFile = File(...)):
    """
    Phân tích commit từ file CSV
    
    Args:
        file: File CSV có cột 'message' hoặc 'commit_message'
    
    Returns:
        {"filename": string, "total": int, "critical": int}
    """
    try:
        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        
        # Đọc file CSV
        df = pd.read_csv(tmp_path)
        tmp_path.unlink()  # Xóa file tạm
        
        # Kiểm tra cột message
        message_col = 'message' if 'message' in df.columns else 'commit_message'
        if message_col not in df.columns:
            raise HTTPException(status_code=400, detail="File thiếu cột 'message' hoặc 'commit_message'")
        
        # Phân tích
        results = {
            "filename": file.filename,
            "total": len(df),
            "critical": 0,
            "sample_results": []
        }
        
        df['is_critical'] = df[message_col].apply(predict_commit)
        results["critical"] = int(df['is_critical'].sum())
        
        # Lấy 5 kết quả mẫu
        sample = df.head(5).to_dict('records')
        results["sample_results"] = [{
            "message": row[message_col][:100] + "..." if len(row[message_col]) > 100 else row[message_col],
            "is_critical": bool(row['is_critical'])
        } for row in sample]
        
        return results
        
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(e)}")

# ==================== COMMIT DIFF ENDPOINTS ====================

@router.get("/{owner}/{repo}/commits/{sha}/diff")
async def get_commit_diff_endpoint(
    owner: str, 
    repo: str, 
    sha: str,
    authorization: str = Header(..., alias="Authorization")
):
    """
    Lấy diff của commit cụ thể
    
    Args:
        owner: Owner của repository
        repo: Tên repository
        sha: SHA của commit
        authorization: GitHub token
        
    Returns:
        Diff content của commit
    """
    try:
        from services.commit_service import get_commit_diff
        
        # Extract token from authorization header
        github_token = None
        if authorization.startswith("Bearer "):
            github_token = authorization.replace("Bearer ", "")
        elif authorization.startswith("token "):
            github_token = authorization.replace("token ", "")
        
        diff = await get_commit_diff(owner, repo, sha, github_token)
        
        if diff is None:
            raise HTTPException(status_code=404, detail="Commit diff not found")
        
        return {
            "sha": sha,
            "repository": f"{owner}/{repo}",
            "diff": diff,
            "size": len(diff),
            "lines_count": len(diff.split('\n')) if diff else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving diff: {str(e)}")

@router.get("/{owner}/{repo}/commits/{sha}/files")
async def get_commit_files_endpoint(
    owner: str, 
    repo: str, 
    sha: str,
    authorization: str = Header(..., alias="Authorization")
):
    """
    Lấy danh sách files thay đổi trong commit
    
    Args:
        owner: Owner của repository
        repo: Tên repository
        sha: SHA của commit
        authorization: GitHub token
        
    Returns:
        Danh sách files thay đổi với stats
    """
    try:
        from services.commit_service import get_commit_files
        
        # Extract token from authorization header
        github_token = None
        if authorization.startswith("Bearer "):
            github_token = authorization.replace("Bearer ", "")
        elif authorization.startswith("token "):
            github_token = authorization.replace("token ", "")
        
        files = await get_commit_files(owner, repo, sha, github_token)
        
        # Calculate summary stats
        total_additions = sum(f.get('additions', 0) for f in files)
        total_deletions = sum(f.get('deletions', 0) for f in files)
        total_changes = sum(f.get('changes', 0) for f in files)
        
        return {
            "sha": sha,
            "repository": f"{owner}/{repo}",
            "files": files,
            "summary": {
                "files_count": len(files),
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "total_changes": total_changes
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")

@router.get("/{owner}/{repo}/commits/{sha}/stats")
async def get_commit_stats_endpoint(
    owner: str, 
    repo: str, 
    sha: str,
    authorization: str = Header(..., alias="Authorization")
):
    """
    Lấy thống kê tổng hợp của commit
    
    Args:
        owner: Owner của repository
        repo: Tên repository
        sha: SHA của commit
        authorization: GitHub token
        
    Returns:
        Thống kê commit bao gồm files và diff
    """
    try:
        from services.commit_service import get_commit_diff, get_commit_files
        
        # Extract token from authorization header
        github_token = None
        if authorization.startswith("Bearer "):
            github_token = authorization.replace("Bearer ", "")
        elif authorization.startswith("token "):
            github_token = authorization.replace("token ", "")
        
        # Get both files and diff
        files = await get_commit_files(owner, repo, sha, github_token)
        diff = await get_commit_diff(owner, repo, sha, github_token)
        
        if not files and not diff:
            raise HTTPException(status_code=404, detail="Commit not found")
        
        # Calculate stats
        total_additions = sum(f.get('additions', 0) for f in files)
        total_deletions = sum(f.get('deletions', 0) for f in files)
        total_changes = sum(f.get('changes', 0) for f in files)
        
        # File type analysis
        file_types = {}
        for file in files:
            filename = file.get('filename', '')
            if '.' in filename:
                ext = filename.split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "sha": sha,
            "repository": f"{owner}/{repo}",
            "stats": {
                "files_count": len(files),
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "total_changes": total_changes,
                "diff_size": len(diff) if diff else 0,
                "diff_lines": len(diff.split('\n')) if diff else 0
            },
            "file_types": file_types,
            "has_diff": diff is not None,
            "has_files": len(files) > 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving commit stats: {str(e)}")