import requests

url = "http://localhost:8000/api/multifusion-commit-analysis/{repo_id}/commits/all/analysis"

# Test không truyền branch_name
# resp1 = requests.get(url.format(repo_id=8))
# print("Không truyền branch_name:", resp1.json())

# Test truyền branch_name
params = {"branch_name": "CommitAnalyst"}
resp2 = requests.get(url.format(repo_id=8), params=params)
print("Truyền branch_name=CommitAnalyst:", resp2.json())