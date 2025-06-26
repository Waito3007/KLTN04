"""
Script thu thập 100.000 commit từ GitHub.
"""
import os
import sys
import time
import json
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import từ các module khác
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_collection.github_collector import GitHubDataCollector

def collect_100k_commits(github_token, output_dir="data", target_commits=100000):
    """
    Thu thập 100.000 commit từ GitHub.
    
    Args:
        github_token: GitHub API token
        output_dir: Thư mục lưu dữ liệu
        target_commits: Số lượng commit mục tiêu
    """
    collector = GitHubDataCollector(token=github_token)
    
    # Danh sách các repo phổ biến
    repos = [
        # Các dự án lớn với lịch sử commit dài
        "torvalds/linux",           # Hàng trăm nghìn commit
        "microsoft/vscode",         # Hàng chục nghìn commit 
        "facebook/react",           # Hàng chục nghìn commit
        "tensorflow/tensorflow",    # Hàng chục nghìn commit
        "angular/angular",          # Hàng chục nghìn commit
        "kubernetes/kubernetes",    # Hàng chục nghìn commit
        "django/django",            # Hàng chục nghìn commit
        "laravel/laravel",          # Hàng chục nghìn commit
        "ruby/ruby",                # Hàng chục nghìn commit
        "rails/rails",              # Hàng chục nghìn commit
        "vuejs/vue",                # Hàng nghìn commit
        "pytorch/pytorch",          # Hàng nghìn commit
        "flutter/flutter",          # Hàng nghìn commit
        "opencv/opencv",            # Hàng nghìn commit
        "pandas-dev/pandas",        # Hàng nghìn commit
        "golang/go",                # Hàng nghìn commit
        "docker/docker-ce",         # Hàng nghìn commit
        "nodejs/node",              # Hàng nghìn commit
        "rust-lang/rust",           # Hàng nghìn commit
        "sveltejs/svelte",          # Commit mới hơn
    ]
    
    # Đường dẫn đầy đủ cho thư mục đầu ra
    full_output_dir = os.path.join(current_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Chiến lược thu thập: chia thành nhiều file để tránh file quá lớn
    commits_per_file = 20000
    commits_per_repo = target_commits // len(repos) + 1  # Phân phối đều giữa các repo
    
    num_files = (target_commits + commits_per_file - 1) // commits_per_file
    
    total_collected = 0
    current_file_commits = []
    file_count = 1
    
    print(f"Bắt đầu thu thập {target_commits} commit từ {len(repos)} repositories...")
    print(f"Dữ liệu sẽ được chia thành khoảng {num_files} file, mỗi file tối đa {commits_per_file} commit")
    
    try:
        for repo_idx, repo in enumerate(repos):
            print(f"\n=== Đang thu thập từ repo [{repo_idx+1}/{len(repos)}]: {repo} ===")
            
            # Số commit cần lấy từ repo này
            commits_to_collect = min(commits_per_repo, target_commits - total_collected)
            if commits_to_collect <= 0:
                break
                
            try:
                # Lấy commit history
                commits = collector.get_commit_history(
                    repo, 
                    max_pages=(commits_to_collect // 100) + 1,
                    per_page=100
                )
                
                # Giới hạn số lượng commit
                commits = commits[:commits_to_collect]
                
                for commit_idx, commit in enumerate(commits):
                    try:
                        commit_sha = commit.get('sha')
                        if commit_sha:
                            # Hiển thị tiến độ
                            progress = (total_collected / target_commits) * 100
                            print(f"\rTiến độ: {progress:.2f}% - Repo {repo_idx+1}/{len(repos)}, " 
                                  f"Commit {commit_idx+1}/{len(commits)} - Tổng: {total_collected}", end="")
                            
                            # Lấy chi tiết commit
                            commit_detail = collector.get_commit_details(repo, commit_sha)
                            if commit_detail:
                                commit_data = collector.extract_commit_data(commit_detail)
                                commit_data['metadata']['repository'] = repo
                                current_file_commits.append(commit_data)
                                total_collected += 1
                                
                                # Nếu đạt đủ số lượng cho một file hoặc đã đủ target, lưu lại
                                if len(current_file_commits) >= commits_per_file or total_collected >= target_commits:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_file = os.path.join(full_output_dir, f"github_commits_part{file_count}_{timestamp}.json")
                                    
                                    # Lưu file
                                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                                    with open(output_file, 'w', encoding='utf-8') as f:
                                        json.dump({
                                            'metadata': {
                                                'total_samples': len(current_file_commits),
                                                'created_at': datetime.now().isoformat(),
                                                'part': file_count,
                                                'repositories': [repo]
                                            },
                                            'data': current_file_commits
                                        }, f, ensure_ascii=False)
                                    
                                    print(f"\nĐã lưu {len(current_file_commits)} commit vào file {output_file}")
                                    current_file_commits = []
                                    file_count += 1
                            
                            # Nếu đạt target, dừng lại
                            if total_collected >= target_commits:
                                print(f"\nĐã đạt target {target_commits} commit. Dừng thu thập.")
                                return
                            
                            # Đợi để tránh rate limit
                            time.sleep(0.2)
                    
                    except Exception as e:
                        print(f"\nLỗi khi xử lý commit {commit_idx} từ {repo}: {str(e)}")
                        # Tiếp tục với commit tiếp theo
                        continue
                
                print(f"\nĐã thu thập {len(commits)} commit từ {repo}")
                
            except Exception as e:
                print(f"\nLỗi khi thu thập từ repo {repo}: {str(e)}")
                # Tiếp tục với repo tiếp theo
                continue
                
    except KeyboardInterrupt:
        print("\nThu thập bị ngắt bởi người dùng.")
    except Exception as e:
        print(f"\nLỗi không xác định: {str(e)}")
    finally:
        # Lưu phần còn lại nếu có
        if current_file_commits:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(full_output_dir, f"github_commits_part{file_count}_{timestamp}.json")
            
            # Lưu file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'total_samples': len(current_file_commits),
                        'created_at': datetime.now().isoformat(),
                        'part': file_count,
                        'repositories': [repo]
                    },
                    'data': current_file_commits
                }, f, ensure_ascii=False)
            
            print(f"\nĐã lưu {len(current_file_commits)} commit còn lại vào file {output_file}")
        
        print(f"\nKết thúc thu thập. Tổng số commit đã thu thập: {total_collected}")


if __name__ == "__main__":
    # Nhập GitHub token
    github_token = input("Nhập GitHub token của bạn: ").strip()
    
    # Xác nhận thực hiện
    print("\nChuẩn bị thu thập 100.000 commit từ GitHub. Quá trình này có thể mất nhiều giờ.")
    print("Các file dữ liệu sẽ được lưu trong thư mục 'data'.")
    confirmation = input("Bạn có muốn tiếp tục không? (y/n): ").strip().lower()
    
    if confirmation == 'y':
        collect_100k_commits(github_token)
    else:
        print("Đã hủy thu thập dữ liệu.")
