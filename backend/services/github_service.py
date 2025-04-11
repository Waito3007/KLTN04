# backend/services/github_service.py
import httpx
import os
from dotenv import load_dotenv

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
