import requests
import json

API_URL = "http://localhost:8000/api/{repo_id}/commits/all/analysis"
REPO_ID = 8
BRANCH_NAME = "CommitAnalyst"

params = {
    "limit": 10,
    "offset": 0,
    "branch_name": BRANCH_NAME
}

url = API_URL.format(repo_id=REPO_ID)

response = requests.get(url, params=params)

print(f"Status code: {response.status_code}")
try:
    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if data.get("commits"):
        print("\nCác thông số numeric features của từng commit:")
        for commit in data["commits"]:
            print(f"SHA: {commit.get('sha')}")
            for k, v in commit.items():
                if k.startswith("num_") or k.endswith("_files") or k in ["num_dirs_changed"]:
                    print(f"  {k}: {v}")
            print("-")
except Exception as e:
    print("Không thể parse JSON hoặc lỗi:", e)
