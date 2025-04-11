# backend/services/github_service.py
import httpx
import os
from dotenv import load_dotenv
from typing import Optional
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com"

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

async def fetch_from_github(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}{url}", headers=headers)
        response.raise_for_status()
        return response.json()



async def fetch_commits(token: str, owner: str, name: str, branch: str, since: Optional[str], until: Optional[str]):
    url = f"https://api.github.com/repos/{owner}/{name}/commits"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    params = {
        "sha": branch
    }
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    async with httpx.AsyncClient() as client:
        res = await client.get(url, headers=headers, params=params)
        res.raise_for_status()
        return res.json()
