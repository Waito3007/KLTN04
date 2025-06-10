from db.models.branches import branches
from db.database import database
from fastapi import HTTPException, Request
import httpx
import logging

logger = logging.getLogger(__name__)

async def get_repo_id_by_owner_and_name(owner: str, repo: str):
    # Placeholder function for getting repository ID by owner and name
    pass

async def save_branch(branch_data):
    query = branches.insert().values(
        name=branch_data["name"],
        repo_id=branch_data["repo_id"],
    )
    await database.execute(query)

async def save_branches(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/branches"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        branches = resp.json()
        logger.info(f"Branches data: {branches}")

    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    for branch in branches:
        branch_data = {
            "name": branch["name"],
            "repo_id": repo_id,
        }
        await save_branch(branch_data)