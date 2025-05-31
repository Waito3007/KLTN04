# backend/ai/modelAi/fetch_from_github_url.py
import os
import re
import json
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv  # <<< ĐÃ THÊM: Import thư viện dotenv

# Tải các biến môi trường từ file .env ở thư mục gốc
load_dotenv()  # <<< ĐÃ THÊM: Gọi hàm để tải file .env

# URL cơ sở của GitHub API
GITHUB_API_BASE_URL = "https://api.github.com"  # <<< ĐÃ SỬA: Sửa lại URL đúng của API

def parse_github_url(url: str) -> tuple[str | None, str | None]:
    """
    Phân tích URL GitHub để lấy tên chủ sở hữu (owner) và tên repository (repo).
    Ví dụ: "https://github.com/facebook/react" -> ("facebook", "react")
    """
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1), match.group(2).replace(".git", "")
    return None, None

def get_commit_details(commit_url: str, headers: dict) -> dict | None:
    """
    Lấy thông tin chi tiết của một commit, bao gồm số liệu thống kê về thay đổi file.
    """
    try:
        response = requests.get(commit_url, headers=headers)
        response.raise_for_status()  # Ném lỗi nếu request thất bại
        return response.json()
    except requests.RequestException as e:
        print(f"Lỗi khi lấy chi tiết commit {commit_url}: {e}")
        return None

def fetch_github_commits(owner: str, repo: str, token: str, max_pages: int = 0):
    """
    Lấy tất cả các commit từ một repository trên GitHub và định dạng chúng.
    
    Args:
        owner (str): Chủ sở hữu repository.
        repo (str): Tên repository.
        token (str): GitHub Personal Access Token để xác thực.
        max_pages (int): Số lượng trang tối đa để lấy (0 = lấy tất cả).
    """
    # <<< ĐÃ SỬA: Sử dụng GITHUB_API_BASE_URL đã được định nghĩa đúng ở trên
    commits_url = f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/commits"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {"per_page": 100}  # Lấy 100 commit mỗi trang

    all_commits_data = []
    page_count = 0

    while commits_url:
        try:
            print(f"Đang lấy dữ liệu từ trang: {commits_url}")
            response = requests.get(commits_url, headers=headers, params=params)
            response.raise_for_status()
            commits = response.json()

            if not commits:
                break

            for commit in commits:
                # Lấy thông tin chi tiết để có stats (insertions, deletions)
                commit_details = get_commit_details(commit['url'], headers)
                if not commit_details:
                    continue

                stats = commit_details.get("stats", {"additions": 0, "deletions": 0})
                files_changed = len(commit_details.get("files", []))

                # Chuyển đổi date string sang định dạng chuẩn
                date_str = commit['commit']['author']['date']
                commit_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d %H:%M:%S')

                # Tạo cấu trúc dữ liệu giống với file `generate_data.py`
                formatted_commit = {
                    "id": commit['sha'],
                    "data_type": "commit_message",
                    "raw_text": commit['commit']['message'],
                    "source_info": {
                        "repo_id": f"{owner}/{repo}", # Sử dụng tên repo thay cho ID
                        "sha": commit['sha'],
                        "author_name": commit['commit']['author']['name'],
                        "author_email": commit['commit']['author']['email'],
                        "date": commit_date,
                        "insertions": stats['additions'],
                        "deletions": stats['deletions'],
                        "files_changed": files_changed
                    },
                    "labels": {
                        "purpose": None,
                        "suspicious": None,
                        "tech_tag": None,
                        "sentiment": None
                    }
                }
                all_commits_data.append(formatted_commit)

            # Lấy URL của trang tiếp theo từ header 'Link'
            commits_url = response.links.get('next', {}).get('url')
            page_count += 1
            if max_pages > 0 and page_count >= max_pages:
                print(f"Đã đạt giới hạn {max_pages} trang.")
                break

        except requests.RequestException as e:
            print(f"Lỗi khi gọi GitHub API: {e}")
            break
            
    return all_commits_data

def save_to_json(data: list, output_path: str):
    """Lưu dữ liệu vào file JSON."""
    try:
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu thành công {len(data)} commits vào file: {output_path}")
    except IOError as e:
        print(f"Lỗi khi ghi file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lấy dữ liệu commit từ một GitHub repository.")
    parser.add_argument("repo_url", type=str, help="URL của GitHub repository (ví dụ: https://github.com/owner/repo).")
    parser.add_argument("--output", type=str, default="backend/ai/collected_data/commit_messages_raw.json", help="Đường dẫn file JSON để lưu kết quả.")
    parser.add_argument("--token", type=str, help="GitHub Personal Access Token.")
    parser.add_argument("--max-pages", type=int, default=0, help="Số trang commit tối đa cần lấy (0 để lấy tất cả).")

    args = parser.parse_args()

    # Ưu tiên lấy token từ argument, nếu không có thì lấy từ biến môi trường
    github_token = args.token or os.getenv("GITHUB_TOKEN")

    if not github_token:
        print("Lỗi: Vui lòng cung cấp GitHub token qua argument --token hoặc biến môi trường GITHUB_TOKEN.")
    else:
        owner, repo = parse_github_url(args.repo_url)
        if owner and repo:
            print(f"Bắt đầu lấy dữ liệu cho repository: {owner}/{repo}")
            all_commits = fetch_github_commits(owner, repo, github_token, args.max_pages)
            if all_commits:
                save_to_json(all_commits, args.output)
        else:
            print(f"Lỗi: URL repository không hợp lệ: {args.repo_url}")