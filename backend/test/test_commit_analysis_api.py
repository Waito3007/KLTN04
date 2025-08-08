
import requests
import json

API_URL = "http://localhost:8000/api/multifusion-commit-analysis/{repo_id}/commits/all/analysis"
REPO_ID = 8
BRANCH_NAME = "CommitAnalyst"

params = {
    "limit": 10,
    "offset": 0,
    "branch_name": BRANCH_NAME
}

url = API_URL.format(repo_id=REPO_ID)

response = requests.get(url, params=params)

result_path = r"backend\test\test_commit_analysis_api_result.txt"
with open(result_path, "w", encoding="utf-8") as f:
    f.write(f"Status code: {response.status_code}\n")
    try:
        data = response.json()
        commits = data.get("commits", [])
        if commits:
            f.write("Snapshot JSON của một số commit đầu và cuối:\n")
            # In ra commit đầu tiên
            f.write("\n--- Mẫu đầu tiên ---\n")
            first = commits[0]
            f.write(json.dumps(first, indent=2, ensure_ascii=False) + "\n")
            # Nếu có nhiều hơn 2 commit, in thêm commit cuối cùng
            if len(commits) > 1:
                f.write("\n--- Mẫu cuối cùng ---\n")
                last = commits[-1]
                f.write(json.dumps(last, indent=2, ensure_ascii=False) + "\n")
        else:
            f.write("Không có commit nào trong response.\n")
    except Exception as e:
        f.write(f"Không thể parse JSON hoặc lỗi: {e}\n")
print(f"Đã ghi kết quả ra file {result_path}")
