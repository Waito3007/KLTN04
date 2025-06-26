"""
Script thu thập dữ liệu commit từ GitHub sử dụng xử lý song song.

Script này tối ưu hóa quá trình thu thập dữ liệu lớn bằng cách:
1. Sử dụng nhiều luồng xử lý song song
2. Hỗ trợ nhiều GitHub token để tăng rate limit
3. Tự động phân chia repository thành các batch
4. Loại bỏ commit trùng lặp
5. Tự động khôi phục khi bị gián đoạn
"""
import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional

# Thêm thư mục gốc vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_collection.parallel_github_collector import ParallelGitHubCollector, split_repos_into_batches

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, "collect_parallel.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_repositories_from_file(file_path: str) -> List[str]:
    """
    Đọc danh sách repositories từ file.
    
    Args:
        file_path: Đường dẫn đến file danh sách repository
        
    Returns:
        Danh sách repository
    """
    repos = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                repo = line.strip()
                if repo and not repo.startswith('#'):
                    repos.append(repo)
        logger.info(f"Đã đọc {len(repos)} repositories từ {file_path}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file repositories: {str(e)}")
    
    return repos

def read_tokens_from_file(file_path: str) -> List[str]:
    """
    Đọc danh sách GitHub token từ file.
    
    Args:
        file_path: Đường dẫn đến file danh sách token
        
    Returns:
        Danh sách token
    """
    tokens = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token and not token.startswith('#'):
                    tokens.append(token)
        logger.info(f"Đã đọc {len(tokens)} token từ {file_path}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file tokens: {str(e)}")
    
    return tokens

def create_default_repository_list(output_file: str) -> None:
    """
    Tạo file danh sách repository mặc định.
    
    Args:
        output_file: Đường dẫn đến file đầu ra
    """
    # Danh sách các repository phổ biến và lớn
    default_repos = [
        # Các dự án phổ biến với nhiều commit
        "torvalds/linux",
        "chromium/chromium",
        "microsoft/vscode",
        "facebook/react",
        "tensorflow/tensorflow",
        "angular/angular",
        "kubernetes/kubernetes",
        "django/django",
        "laravel/laravel",
        "ruby/ruby",
        "rails/rails",
        "vuejs/vue",
        "pytorch/pytorch",
        "flutter/flutter",
        "opencv/opencv",
        "pandas-dev/pandas",
        "golang/go",
        "docker/docker",
        "nodejs/node",
        "rust-lang/rust",
        
        # Thêm các dự án phổ biến khác
        "microsoft/TypeScript",
        "apple/swift",
        "facebook/react-native",
        "vercel/next.js",
        "puppeteer/puppeteer",
        "expressjs/express",
        "spring-projects/spring-boot",
        "elastic/elasticsearch",
        "bitcoin/bitcoin",
        "ethereum/go-ethereum",
        "godotengine/godot",
        "python/cpython",
        "scikit-learn/scikit-learn",
        "tensorflow/models",
        "facebook/create-react-app",
        "openai/openai-python",
        "microsoft/PowerToys",
        "huggingface/transformers",
        "hashicorp/terraform",
        "denoland/deno"
    ]
    
    # Ghi file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Danh sách repositories để thu thập dữ liệu\n")
        f.write("# Mỗi dòng là một repository theo format: owner/repo\n")
        f.write("# Dòng bắt đầu bằng # sẽ bị bỏ qua\n\n")
        
        for repo in default_repos:
            f.write(f"{repo}\n")
    
    logger.info(f"Đã tạo file danh sách repository mặc định: {output_file}")

def create_token_file(output_file: str) -> None:
    """
    Tạo file token mẫu.
    
    Args:
        output_file: Đường dẫn đến file đầu ra
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Danh sách GitHub token\n")
        f.write("# Mỗi dòng là một token\n")
        f.write("# Dòng bắt đầu bằng # sẽ bị bỏ qua\n\n")
        f.write("# Thêm token của bạn vào đây:\n")
        f.write("ghp_YourGitHubPersonalAccessToken1\n")
        f.write("ghp_YourGitHubPersonalAccessToken2\n")
    
    logger.info(f"Đã tạo file token mẫu: {output_file}")

def create_state_file(state_file: str, total_collected: int = 0, 
                     current_batch: int = 0, current_repo_index: int = 0,
                     repositories: List[str] = None, batch_size: int = 5) -> None:
    """
    Tạo hoặc cập nhật file trạng thái.
    
    Args:
        state_file: Đường dẫn đến file trạng thái
        total_collected: Tổng số commit đã thu thập
        current_batch: Batch hiện tại
        current_repo_index: Chỉ số repository hiện tại
        repositories: Danh sách repository
        batch_size: Kích thước mỗi batch
    """
    repositories = repositories or []
    batches = split_repos_into_batches(repositories, batch_size)
    
    state = {
        "total_collected": total_collected,
        "current_batch": current_batch,
        "current_repo_index": current_repo_index,
        "started_at": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "repositories": repositories,
        "batch_size": batch_size,
        "total_batches": len(batches),
        "remaining_repos": len(repositories) - current_repo_index
    }
    
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"Đã cập nhật file trạng thái: {state_file}")

def collect_commits_parallel(
    github_tokens: List[str],
    output_dir: str,
    repositories: List[str] = None,
    repo_file: str = None,
    max_commits_per_repo: int = 1000,
    max_workers: int = 4,
    batch_size: int = 5,
    deduplicate: bool = True
) -> None:
    """
    Thu thập commit từ GitHub sử dụng xử lý song song.
    
    Args:
        github_tokens: Danh sách GitHub token
        output_dir: Thư mục đầu ra
        repositories: Danh sách repository (nếu không cung cấp, sẽ đọc từ repo_file)
        repo_file: File chứa danh sách repository
        max_commits_per_repo: Số lượng commit tối đa từ mỗi repository
        max_workers: Số lượng worker song song
        batch_size: Kích thước mỗi batch repository
        deduplicate: Có loại bỏ commit trùng lặp không
    """
    # Xác định danh sách repository
    if not repositories and repo_file:
        repositories = read_repositories_from_file(repo_file)
    
    if not repositories:
        logger.error("Không có repositories nào được cung cấp")
        return
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Kiểm tra file trạng thái
    state_file = os.path.join(output_dir, "parallel_collection_state.json")
    current_batch = 0
    current_repo_index = 0
    total_collected = 0
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            current_batch = state.get("current_batch", 0)
            current_repo_index = state.get("current_repo_index", 0)
            total_collected = state.get("total_collected", 0)
            stored_batch_size = state.get("batch_size", batch_size)
            
            # Đảm bảo kích thước batch nhất quán
            if batch_size != stored_batch_size:
                logger.warning(f"Kích thước batch đã thay đổi từ {stored_batch_size} thành {batch_size}. Đang sử dụng giá trị mới.")
            
            logger.info(f"Tiếp tục từ batch {current_batch}, repository {current_repo_index}, đã thu thập {total_collected} commit")
        except Exception as e:
            logger.error(f"Lỗi khi đọc file trạng thái: {str(e)}")
    
    # Tạo file trạng thái nếu chưa tồn tại
    create_state_file(
        state_file=state_file,
        total_collected=total_collected,
        current_batch=current_batch,
        current_repo_index=current_repo_index,
        repositories=repositories,
        batch_size=batch_size
    )
    
    # Tạo collector
    collector = ParallelGitHubCollector(token_list=github_tokens)
    
    # Chia repositories thành các batch
    all_batches = split_repos_into_batches(repositories[current_repo_index:], batch_size)
    
    # Bắt đầu thu thập
    logger.info(f"Bắt đầu thu thập dữ liệu từ {len(repositories[current_repo_index:])} repositories, chia thành {len(all_batches)} batch")
    
    try:
        for i, batch in enumerate(all_batches, start=current_batch):
            batch_start_index = i * batch_size + current_repo_index
            batch_end_index = min(batch_start_index + len(batch), len(repositories))
            batch_repos = repositories[batch_start_index:batch_end_index]
            
            logger.info(f"Xử lý batch {i+1}/{len(all_batches)+current_batch} với {len(batch_repos)} repositories")
            
            # Tạo đường dẫn đầu ra cho batch này
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_output = os.path.join(output_dir, f"github_commits_batch{i+1}_{timestamp}.json")
            
            # Thu thập dữ liệu
            if deduplicate:
                collector.collect_with_deduplication(
                    repos=batch_repos,
                    output_path=batch_output,
                    max_commits_per_repo=max_commits_per_repo,
                    max_workers=max_workers
                )
            else:
                collector.collect_parallel(
                    repos=batch_repos,
                    output_path=batch_output,
                    max_commits_per_repo=max_commits_per_repo,
                    max_workers=max_workers
                )
            
            # Đếm số commit đã thu thập
            try:
                with open(batch_output, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                batch_collected = len(data.get('data', []))
                total_collected += batch_collected
                logger.info(f"Batch {i+1} đã thu thập được {batch_collected} commit")
            except Exception as e:
                logger.error(f"Lỗi khi đọc file kết quả: {str(e)}")
            
            # Cập nhật trạng thái
            create_state_file(
                state_file=state_file,
                total_collected=total_collected,
                current_batch=i+1,
                current_repo_index=batch_end_index,
                repositories=repositories,
                batch_size=batch_size
            )
        
        logger.info(f"Đã hoàn thành thu thập tất cả {len(all_batches)} batch")
        logger.info(f"Tổng cộng đã thu thập {total_collected} commit")
        
    except KeyboardInterrupt:
        logger.info("Thu thập bị ngắt bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình thu thập: {str(e)}")
    finally:
        # Cập nhật trạng thái lần cuối
        create_state_file(
            state_file=state_file,
            total_collected=total_collected,
            current_batch=current_batch,
            current_repo_index=current_repo_index,
            repositories=repositories,
            batch_size=batch_size
        )

def merge_commit_files(input_dir: str, output_file: str, deduplicate: bool = True) -> None:
    """
    Kết hợp nhiều file commit thành một file duy nhất.
    
    Args:
        input_dir: Thư mục chứa các file commit
        output_file: Đường dẫn file đầu ra
        deduplicate: Có loại bỏ commit trùng lặp không
    """
    logger.info(f"Bắt đầu kết hợp các file commit từ {input_dir}")
    
    all_commits = []
    all_repos = set()
    seen_commit_ids = set()
    total_files = 0
    
    # Tìm tất cả các file JSON chứa commit
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith("github_commits_") and file.endswith(".json"):
                total_files += 1
    
    if total_files == 0:
        logger.warning(f"Không tìm thấy file commit nào trong {input_dir}")
        return
    
    logger.info(f"Tìm thấy {total_files} file commit để kết hợp")
    
    # Đọc từng file và kết hợp dữ liệu
    file_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith("github_commits_") and file.endswith(".json"):
                file_path = os.path.join(root, file)
                file_count += 1
                
                logger.info(f"[{file_count}/{total_files}] Đang đọc file {file}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Lấy danh sách commit
                    if 'data' in data:
                        commits = data['data']
                        
                        # Loại bỏ trùng lặp nếu cần
                        if deduplicate:
                            unique_commits = []
                            for commit in commits:
                                commit_id = commit.get('metadata', {}).get('commit_id', '')
                                if commit_id and commit_id not in seen_commit_ids:
                                    seen_commit_ids.add(commit_id)
                                    unique_commits.append(commit)
                            
                            all_commits.extend(unique_commits)
                            logger.info(f"  -> Đã thêm {len(unique_commits)}/{len(commits)} commit duy nhất")
                        else:
                            all_commits.extend(commits)
                            logger.info(f"  -> Đã thêm {len(commits)} commit")
                    
                    # Lấy danh sách repo
                    if 'metadata' in data and 'repositories' in data['metadata']:
                        repos = data['metadata']['repositories']
                        if isinstance(repos, list):
                            all_repos.update(repos)
                    
                except Exception as e:
                    logger.error(f"  -> Lỗi khi đọc file {file}: {str(e)}")
    
    if deduplicate:
        logger.info(f"Đã đọc xong {file_count} file, tổng cộng {len(all_commits)} commit duy nhất")
    else:
        logger.info(f"Đã đọc xong {file_count} file, tổng cộng {len(all_commits)} commit")
    
    if not all_commits:
        logger.warning("Không có commit nào để kết hợp")
        return
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Lưu file kết hợp
    logger.info(f"Đang lưu {len(all_commits)} commit vào {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_samples': len(all_commits),
                'created_at': datetime.now().isoformat(),
                'repositories': list(all_repos),
                'source_files': file_count,
                'deduplicated': deduplicate
            },
            'data': all_commits
        }, f, ensure_ascii=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"Đã lưu thành công! Kích thước file: {file_size_mb:.2f} MB")

def run_interactive():
    """Chạy chế độ tương tác để hướng dẫn người dùng."""
    print("\n=== Thu thập dữ liệu GitHub commit - Chế độ tương tác ===\n")
    
    # Xác định thư mục data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Tạo file repositories mẫu nếu chưa tồn tại
    repo_file = os.path.join(data_dir, "repositories.txt")
    if not os.path.exists(repo_file):
        print("Không tìm thấy file danh sách repository.")
        create = input("Bạn có muốn tạo file danh sách repository mẫu không? (y/n): ").strip().lower()
        if create == 'y':
            create_default_repository_list(repo_file)
            print(f"Đã tạo file {repo_file}. Vui lòng chỉnh sửa file này trước khi tiếp tục.")
            print("Bạn có thể thêm hoặc xóa các repository theo nhu cầu.")
            return
    
    # Tạo file token mẫu nếu chưa tồn tại
    token_file = os.path.join(data_dir, "tokens.txt")
    if not os.path.exists(token_file):
        print("Không tìm thấy file GitHub tokens.")
        create = input("Bạn có muốn tạo file token mẫu không? (y/n): ").strip().lower()
        if create == 'y':
            create_token_file(token_file)
            print(f"Đã tạo file {token_file}. Vui lòng chỉnh sửa file này trước khi tiếp tục.")
            print("Thêm GitHub token của bạn vào file này.")
            return
    
    # Đọc tokens
    github_tokens = read_tokens_from_file(token_file)
    if not github_tokens:
        print("Không tìm thấy GitHub token hợp lệ.")
        token = input("Nhập GitHub token của bạn: ").strip()
        if token:
            github_tokens = [token]
        else:
            print("Cần ít nhất một GitHub token để tiếp tục.")
            return
    
    # Đọc repositories
    repositories = read_repositories_from_file(repo_file)
    if not repositories:
        print(f"Không tìm thấy repository nào trong {repo_file}")
        return
    
    print(f"Đã tìm thấy {len(repositories)} repositories và {len(github_tokens)} GitHub token.")
    
    # Cấu hình thu thập
    try:
        max_commits = int(input("Số lượng commit tối đa từ mỗi repository (mặc định 1000): ") or "1000")
    except ValueError:
        print("Giá trị không hợp lệ, sử dụng mặc định 1000")
        max_commits = 1000
    
    try:
        max_workers = int(input("Số luồng xử lý song song (mặc định 4): ") or "4")
    except ValueError:
        print("Giá trị không hợp lệ, sử dụng mặc định 4")
        max_workers = 4
    
    try:
        batch_size = int(input("Kích thước mỗi batch (mặc định 5 repositories): ") or "5")
    except ValueError:
        print("Giá trị không hợp lệ, sử dụng mặc định 5")
        batch_size = 5
    
    deduplicate = input("Loại bỏ commit trùng lặp? (y/n, mặc định y): ").strip().lower() != 'n'
    
    # Tính toán số lượng commit dự kiến
    estimated_commits = len(repositories) * max_commits
    print(f"\nDự kiến thu thập tối đa {estimated_commits} commit từ {len(repositories)} repositories.")
    print(f"Sử dụng {max_workers} luồng xử lý song song, chia thành các batch {batch_size} repositories.")
    print(f"Loại bỏ trùng lặp: {'Có' if deduplicate else 'Không'}")
    
    # Xác nhận
    confirm = input("\nBạn có chắc chắn muốn bắt đầu thu thập? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Đã hủy thu thập dữ liệu")
        return
    
    # Bắt đầu thu thập
    output_dir = os.path.join(data_dir, "parallel_commits")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nBắt đầu thu thập dữ liệu...")
    print(f"Dữ liệu sẽ được lưu vào: {output_dir}")
    print("Quá trình này có thể mất nhiều thời gian (2-12 giờ tùy thuộc vào số lượng commit)")
    print("Bạn có thể ngắt quá trình bất kỳ lúc nào bằng Ctrl+C, và tiếp tục sau")
    
    try:
        collect_commits_parallel(
            github_tokens=github_tokens,
            output_dir=output_dir,
            repositories=repositories,
            max_commits_per_repo=max_commits,
            max_workers=max_workers,
            batch_size=batch_size,
            deduplicate=deduplicate
        )
        
        # Sau khi thu thập xong, hỏi người dùng có muốn kết hợp không
        merge = input("\nBạn có muốn kết hợp tất cả file commit thành một file duy nhất không? (y/n): ").strip().lower()
        if merge == 'y':
            output_file = os.path.join(data_dir, "all_commits_merged_parallel.json")
            merge_commit_files(output_dir, output_file, deduplicate)
    
    except KeyboardInterrupt:
        print("\nQuá trình thu thập đã bị ngắt bởi người dùng")
        print("Bạn có thể tiếp tục sau bằng cách chạy lại script này")
    except Exception as e:
        print(f"Lỗi khi chạy quá trình thu thập: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Thu thập dữ liệu commit từ GitHub sử dụng xử lý song song')
    subparsers = parser.add_subparsers(dest='command', help='Lệnh')
    
    # Chế độ tương tác
    subparsers.add_parser('interactive', help='Chạy chế độ tương tác')
    
    # Lệnh tạo file
    create_parser = subparsers.add_parser('create-files', help='Tạo các file mẫu')
    create_parser.add_argument('--output_dir', default='../data', help='Thư mục đầu ra')
    
    # Lệnh thu thập
    collect_parser = subparsers.add_parser('collect', help='Thu thập commit')
    collect_parser.add_argument('--token_file', help='Đường dẫn đến file chứa GitHub tokens')
    collect_parser.add_argument('--tokens', nargs='+', help='Danh sách GitHub tokens')
    collect_parser.add_argument('--repo_file', help='Đường dẫn đến file chứa danh sách repository')
    collect_parser.add_argument('--output_dir', default='../data/parallel_commits', help='Thư mục đầu ra')
    collect_parser.add_argument('--max_commits', type=int, default=1000, help='Số commit tối đa từ mỗi repository')
    collect_parser.add_argument('--max_workers', type=int, default=4, help='Số luồng xử lý song song')
    collect_parser.add_argument('--batch_size', type=int, default=5, help='Kích thước mỗi batch repository')
    collect_parser.add_argument('--no_deduplicate', action='store_true', help='Không loại bỏ commit trùng lặp')
    
    # Lệnh kết hợp
    merge_parser = subparsers.add_parser('merge', help='Kết hợp các file commit')
    merge_parser.add_argument('--input_dir', required=True, help='Thư mục chứa các file commit')
    merge_parser.add_argument('--output_file', required=True, help='File đầu ra')
    merge_parser.add_argument('--no_deduplicate', action='store_true', help='Không loại bỏ commit trùng lặp')
    
    args = parser.parse_args()
    
    if args.command == 'interactive':
        run_interactive()
    
    elif args.command == 'create-files':
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        repo_file = os.path.join(output_dir, "repositories.txt")
        token_file = os.path.join(output_dir, "tokens.txt")
        
        create_default_repository_list(repo_file)
        create_token_file(token_file)
    
    elif args.command == 'collect':
        # Đọc tokens
        tokens = []
        if args.tokens:
            tokens = args.tokens
        elif args.token_file:
            tokens = read_tokens_from_file(args.token_file)
        
        if not tokens:
            logger.error("Không có GitHub token nào được cung cấp")
            sys.exit(1)
        
        # Đọc repositories
        repos = None
        if args.repo_file:
            repos = read_repositories_from_file(args.repo_file)
        
        if not repos:
            logger.error("Không có repositories nào được cung cấp")
            sys.exit(1)
        
        # Thu thập dữ liệu
        collect_commits_parallel(
            github_tokens=tokens,
            output_dir=args.output_dir,
            repositories=repos,
            max_commits_per_repo=args.max_commits,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            deduplicate=not args.no_deduplicate
        )
    
    elif args.command == 'merge':
        merge_commit_files(
            input_dir=args.input_dir,
            output_file=args.output_file,
            deduplicate=not args.no_deduplicate
        )
    
    else:
        parser.print_help()
