import json
import os
import logging
import datetime
from pathlib import Path
import hashlib
from typing import Dict, List, Set, Any

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("merge_commits.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_open(file_path, mode="r"):
    """Mở file an toàn với UTF-8, xử lý lỗi encoding."""
    try:
        return open(file_path, mode, encoding="utf-8")
    except Exception as e:
        logger.error(f"Lỗi khi mở file {file_path}: {str(e)}")
        raise

def load_json_file(file_path: str) -> Dict:
    """Đọc file JSON và trả về dữ liệu."""
    try:
        with safe_open(file_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file JSON {file_path}: {str(e)}")
        return {"metadata": {}, "data": []}

def get_commit_hash(commit: Dict) -> str:
    """Tạo hash để xác định commit duy nhất."""
    # Tạo chuỗi từ các trường quan trọng
    key_str = ""
    if "sha" in commit:
        key_str += commit["sha"]
    if "commit" in commit and "message" in commit["commit"]:
        key_str += commit["commit"]["message"]
    if "commit" in commit and "author" in commit["commit"] and "date" in commit["commit"]["author"]:
        key_str += commit["commit"]["author"]["date"]
    
    # Nếu không có thông tin nào, sử dụng toàn bộ commit
    if not key_str and commit:
        key_str = json.dumps(commit, sort_keys=True)
        
    return hashlib.md5(key_str.encode()).hexdigest()

def is_commit_empty(commit: Dict) -> bool:
    """Kiểm tra xem commit có trống hay không."""
    # Commit không có dữ liệu
    if not commit:
        return True
        
    # Kiểm tra commit có cấu trúc cơ bản
    if "commit" not in commit:
        return True
        
    # Kiểm tra commit có message
    if "message" not in commit["commit"]:
        return True
        
    # Kiểm tra message có nội dung
    if not commit["commit"]["message"].strip():
        return True
        
    return False

def merge_commits(input_files: List[str], output_file: str) -> Dict:
    """Gộp nhiều file commit, loại bỏ trùng lặp và làm sạch."""
    all_commits = []
    unique_commits = {}
    stats = {
        "original_total": 0,
        "empty_removed": 0,
        "duplicates_removed": 0,
        "final_count": 0
    }
    repositories = set()
    
    # Đọc từng file và xử lý
    for file_path in input_files:
        logger.info(f"Đang xử lý file {file_path}")
        json_data = load_json_file(file_path)
        
        # Lấy danh sách repositories
        if "metadata" in json_data and "repositories" in json_data["metadata"]:
            for repo in json_data["metadata"]["repositories"]:
                repositories.add(repo)
        
        # Xử lý từng commit
        if "data" in json_data:
            commits = json_data["data"]
            stats["original_total"] += len(commits)
            
            for commit in commits:
                # Kiểm tra commit rỗng
                if is_commit_empty(commit):
                    stats["empty_removed"] += 1
                    continue
                
                # Tạo hash và kiểm tra trùng lặp
                commit_hash = get_commit_hash(commit)
                if commit_hash not in unique_commits:
                    unique_commits[commit_hash] = commit
                else:
                    stats["duplicates_removed"] += 1
    
    # Chuyển từ dict sang list
    all_commits = list(unique_commits.values())
    stats["final_count"] = len(all_commits)
    
    # Tạo metadata cho file đầu ra
    result = {
        "metadata": {
            "total_samples": len(all_commits),
            "created_at": datetime.datetime.now().isoformat(),
            "repositories": list(repositories),
            "is_complete": True,
            "source_files": [os.path.basename(f) for f in input_files],
            "cleaning_stats": stats
        },
        "data": all_commits
    }
    
    # Ghi file đầu ra
    try:
        with safe_open(output_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu file kết quả vào {output_file}")
    except Exception as e:
        logger.error(f"Lỗi khi ghi file kết quả {output_file}: {str(e)}")
    
    return stats

def main():
    # Thư mục chứa các file commit JSON
    input_dir = "e:\\Dự Án Của Nghĩa\\KLTN04\\backend\\ai\\multimodal_fusion_network\\data_collection\\data\\github_commits"
    
    # Thư mục đầu ra
    output_dir = "e:\\Dự Án Của Nghĩa\\KLTN04\\backend\\ai\\multimodal_fusion_network\\data\\processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm các file JSON commit cần xử lý
    input_files = [
        os.path.join(input_dir, "github_commits_batch1_20250625_231618_part0.json"),
        os.path.join(input_dir, "github_commits_batch1_20250625_234019_part0.json"),
        os.path.join(input_dir, "github_commits_batch1_20250625_234019_part1.json")
    ]
    
    # Tên file đầu ra
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"merged_commits_{timestamp}.json")
    
    # Gộp các file
    stats = merge_commits(input_files, output_file)
    
    # In thống kê
    logger.info(f"Kết quả: {stats['original_total']} commit ban đầu")
    logger.info(f"Đã loại bỏ {stats['empty_removed']} commit rỗng")
    logger.info(f"Đã loại bỏ {stats['duplicates_removed']} commit trùng lặp")
    logger.info(f"Còn lại {stats['final_count']} commit trong file kết quả")

if __name__ == "__main__":
    main()