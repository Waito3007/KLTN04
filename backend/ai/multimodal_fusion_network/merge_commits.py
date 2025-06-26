"""
Script kết hợp nhiều file commit thành một file duy nhất.
"""
import os
import sys
import json
import glob
from datetime import datetime

def merge_commit_files(input_dir, output_dir, max_file_size_mb=500):
    """
    Kết hợp nhiều file commit thành một file duy nhất.
    
    Args:
        input_dir: Thư mục chứa các file commit
        output_dir: Thư mục đầu ra sau khi kết hợp
        max_file_size_mb: Kích thước tối đa của mỗi file đầu ra (MB)
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tên file đầu ra cơ bản
    output_base = os.path.join(output_dir, f"merged_commits_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Tìm tất cả file JSON trong thư mục
    json_files = glob.glob(os.path.join(input_dir, "github_commits_*.json"))
    
    if not json_files:
        print(f"Không tìm thấy file JSON nào trong {input_dir}")
        return
    
    print(f"Tìm thấy {len(json_files)} file để kết hợp.")
    
    all_repos = set()
    total_commits = 0
    file_count = 1
    current_commits = []
    max_size_bytes = max_file_size_mb * 1024 * 1024  # Chuyển đổi MB thành bytes
    
    for idx, file_path in enumerate(json_files):
        print(f"Đang xử lý file [{idx+1}/{len(json_files)}]: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Lấy danh sách commit
                commits = data.get('data', [])
                
                # Thêm repo vào danh sách
                if 'metadata' in data and 'repositories' in data['metadata']:
                    repos = data['metadata']['repositories']
                    if isinstance(repos, list):
                        all_repos.update(repos)
                
                # Xử lý từng commit
                for commit in commits:
                    current_commits.append(commit)
                    total_commits += 1
                    
                    # Kiểm tra nếu đã đạt kích thước tối đa, lưu file
                    if len(current_commits) % 1000 == 0:
                        # Ước tính kích thước hiện tại
                        current_size = sys.getsizeof(json.dumps(current_commits))
                        if current_size > max_size_bytes:
                            # Lưu file hiện tại
                            save_merged_file(
                                current_commits, 
                                f"{output_base}_part{file_count}.json",
                                all_repos,
                                file_count
                            )
                            current_commits = []
                            file_count += 1
                
                print(f"  -> Đã thêm {len(commits)} commit.")
                
        except Exception as e:
            print(f"  -> Lỗi khi xử lý file {file_path}: {str(e)}")
    
    # Lưu phần còn lại nếu có
    if current_commits:
        save_merged_file(
            current_commits, 
            f"{output_base}_part{file_count}.json",
            all_repos,
            file_count
        )
    
    print(f"Đã kết hợp tổng cộng {total_commits} commit thành {file_count} file.")
    print(f"Các file đầu ra được lưu trong thư mục: {output_dir}")


def save_merged_file(commits, output_file, repos, part_number):
    """Lưu danh sách commit vào file."""
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Lưu file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_samples': len(commits),
                'created_at': datetime.now().isoformat(),
                'repositories': list(repos),
                'part': part_number
            },
            'data': commits
        }, f, ensure_ascii=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Đã lưu {len(commits)} commit vào {output_file} (Kích thước: {file_size_mb:.2f} MB)")


if __name__ == "__main__":
    # Đường dẫn đến các thư mục
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Thư mục chứa file input
    input_dir = os.path.join(current_dir, "data_collection", "data", "github_commits")
    
    # Thư mục đầu ra
    output_dir = os.path.join(current_dir, "data", "processed")
    
    # Kích thước tối đa mỗi file (MB)
    max_file_size = 500  # 500MB
    
    print(f"Bắt đầu kết hợp các file từ {input_dir}")
    print(f"Mỗi file đầu ra sẽ có kích thước tối đa {max_file_size}MB")
    print(f"File đầu ra sẽ được lưu trong {output_dir}")
    
    merge_commit_files(input_dir, output_dir, max_file_size)
