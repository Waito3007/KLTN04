"""
Script đơn giản để hiển thị thống kê về dữ liệu commit đã thu thập.
Chạy script này để xem thông tin tổng quan về dữ liệu.
"""
import os
import json
import logging
from collections import Counter
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_commit_data(data_dir, file_pattern="github_commits_"):
    """
    Phân tích dữ liệu commit đã thu thập.
    
    Args:
        data_dir: Thư mục chứa dữ liệu
        file_pattern: Mẫu tên file để phân tích
    """
    logger.info(f"Đang phân tích dữ liệu commit trong {data_dir}")
    
    # Tìm các file commit
    commit_files = [f for f in os.listdir(data_dir) if f.startswith(file_pattern) and f.endswith(".json")]
    
    if not commit_files:
        logger.error(f"Không tìm thấy file commit nào trong {data_dir}")
        return
    
    logger.info(f"Tìm thấy {len(commit_files)} file commit để phân tích")
    
    # Thống kê
    total_commits = 0
    repositories = Counter()
    authors = Counter()
    file_types = Counter()
    commit_sizes = {"small": 0, "medium": 0, "large": 0, "very_large": 0}
    commits_by_date = {}
    
    # Phân tích từng file
    for file_name in commit_files:
        file_path = os.path.join(data_dir, file_name)
        logger.info(f"Đang phân tích file {file_name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'data' not in data:
                logger.warning(f"File {file_name} không có trường 'data'")
                continue
            
            commits = data['data']
            total_commits += len(commits)
            
            for commit in commits:
                # Thông tin repository
                repo = commit.get('metadata', {}).get('repository', 'unknown')
                repositories[repo] += 1
                
                # Thông tin tác giả
                author = commit.get('metadata', {}).get('author', 'unknown')
                authors[author] += 1
                
                # Thông tin loại file
                file_types_dict = commit.get('metadata', {}).get('file_types', {})
                for ftype, count in file_types_dict.items():
                    file_types[ftype] += count
                
                # Thống kê kích thước commit
                changes = commit.get('metadata', {}).get('total_changes', 0)
                if changes <= 10:
                    commit_sizes["small"] += 1
                elif changes <= 50:
                    commit_sizes["medium"] += 1
                elif changes <= 200:
                    commit_sizes["large"] += 1
                else:
                    commit_sizes["very_large"] += 1
                
                # Thống kê theo thời gian
                timestamp = commit.get('metadata', {}).get('timestamp', '')
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m')
                        commits_by_date[date] = commits_by_date.get(date, 0) + 1
                    except (ValueError, TypeError):
                        pass
        
        except Exception as e:
            logger.error(f"Lỗi khi phân tích file {file_name}: {str(e)}")
    
    # Hiển thị kết quả
    logger.info("\n=== THỐNG KÊ DỮ LIỆU COMMIT ===")
    logger.info(f"Tổng số commit: {total_commits}")
    
    logger.info("\n--- Top 10 Repository ---")
    for repo, count in repositories.most_common(10):
        logger.info(f"{repo}: {count} commits ({count/total_commits*100:.2f}%)")
    
    logger.info("\n--- Top 10 Tác giả ---")
    for author, count in authors.most_common(10):
        logger.info(f"{author}: {count} commits ({count/total_commits*100:.2f}%)")
    
    logger.info("\n--- Top 10 Loại file ---")
    for ftype, count in file_types.most_common(10):
        logger.info(f"{ftype}: {count} files")
    
    logger.info("\n--- Phân bố kích thước commit ---")
    for size, count in commit_sizes.items():
        logger.info(f"{size}: {count} commits ({count/total_commits*100:.2f}%)")
    
    logger.info("\n--- Phân bố theo thời gian ---")
    for date in sorted(commits_by_date.keys())[-10:]:  # 10 tháng gần nhất
        logger.info(f"{date}: {commits_by_date[date]} commits")
    
    logger.info("\n=== KẾT THÚC THỐNG KÊ ===")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phân tích dữ liệu commit')
    parser.add_argument('--data_dir', default=None, help='Thư mục chứa dữ liệu commit')
    
    args = parser.parse_args()
    
    # Xác định thư mục dữ liệu
    if args.data_dir:
        data_dir = args.data_dir
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    if not os.path.exists(data_dir):
        logger.error(f"Thư mục dữ liệu {data_dir} không tồn tại!")
    else:
        analyze_commit_data(data_dir)
