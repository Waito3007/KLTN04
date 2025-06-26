"""
Script thu thập 100.000 commit từ GitHub.
"""
import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# Thêm thư mục gốc vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_collection.github_collector import RateLimitExceededException, GitHubDataCollector

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, "collect_100k.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_100k_commits(github_token: str = None, token_file: str = None, output_dir: str = "../data/github_commits", target_commits: int = 100000):
    """
    Thu thập khoảng 100.000 commit từ các repository phổ biến.
    
    Args:
        github_token: GitHub API token (có thể là một token hoặc danh sách các token)
        token_file: Đường dẫn đến file chứa danh sách token
        output_dir: Thư mục để lưu dữ liệu
        target_commits: Số lượng commit mục tiêu cần thu thập
    """
    # Xử lý token
    tokens = []
    
    # Đọc token từ file nếu có
    if token_file:
        file_tokens = read_tokens_from_file(token_file)
        tokens.extend(file_tokens)
    
    # Thêm token được cung cấp trực tiếp
    if github_token:
        if isinstance(github_token, list):
            tokens.extend(github_token)
        else:
            tokens.append(github_token)
    
    # Nếu không có token nào, thử đọc từ biến môi trường
    if not tokens:
        env_token = os.environ.get('GITHUB_TOKEN')
        if env_token:
            tokens.append(env_token)
            logger.info("Sử dụng token từ biến môi trường GITHUB_TOKEN")
    
    # Kiểm tra xem có token không
    if not tokens:
        logger.error("Không tìm thấy token nào. Vui lòng cung cấp token qua tham số, file token hoặc biến môi trường GITHUB_TOKEN")
        return
    
    logger.info(f"Sẽ sử dụng {len(tokens)} token cho việc thu thập dữ liệu")
    
    # Khởi tạo collector với token đầu tiên
    # (trong tương lai có thể mở rộng để luân chuyển token)
    current_token = tokens[0]
    collector = GitHubDataCollector(token=current_token)
    
    # Danh sách các repo phổ biến với nhiều commit
    repos = [
        # Dự án lớn với lịch sử commit dài
        "torvalds/linux",           # Hàng trăm nghìn commit
        "chromium/chromium",        # Hàng trăm nghìn commit
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
        "docker/docker",            # Hàng nghìn commit
        "nodejs/node",              # Hàng nghìn commit
        "rust-lang/rust",           # Hàng nghìn commit
    ]
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo trạng thái ban đầu
    state_file = os.path.join(output_dir, "collection_state.json")
    if os.path.exists(state_file):
        logger.info(f"Tìm thấy state file, tiếp tục thu thập từ trạng thái trước đó")
        try:
            with safe_open(state_file, 'r') as f:
                state = json.load(f)
        except Exception as e:
            logger.error(f"Lỗi khi đọc state file: {str(e)}")
            logger.info("Tạo state file mới")
            state = {
                "total_collected": 0,
                "current_repo_index": 0,
                "current_commit_index": 0,
                "current_chunk_index": 0,
                "started_at": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
    else:
        state = {
            "total_collected": 0,
            "current_repo_index": 0,
            "current_commit_index": 0,
            "current_chunk_index": 0,
            "started_at": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat()
        }
    
    # Khởi tạo tham số
    total_collected = state["total_collected"]
    current_repo_index = state["current_repo_index"]
    current_commit_index = state["current_commit_index"]
    current_chunk_index = state["current_chunk_index"]
    
    # Số commit cần lấy từ mỗi repo
    commits_per_repo = (target_commits - total_collected) // (len(repos) - current_repo_index) + 1
    logger.info(f"Sẽ thu thập khoảng {commits_per_repo} commit từ mỗi repo còn lại")
    
    try:
        # Thu thập dữ liệu theo từng batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"github_commits_batch{current_chunk_index+1}_{timestamp}.json")
        
        # Thu thập và lưu dữ liệu với khả năng tiếp tục
        collector.collect_and_save_data_with_resume(
            repos=repos,
            output_path=output_file,
            max_commits_per_repo=commits_per_repo,
            save_chunk_size=1000,  # Lưu sau mỗi 1000 commit
            start_from_repo_index=current_repo_index,
            start_from_commit_index=current_commit_index,
            chunk_start_index=current_chunk_index
        )
        
        logger.info(f"Đã hoàn thành thu thập dữ liệu")
        
    except KeyboardInterrupt:
        logger.info("Thu thập bị ngắt bởi người dùng")
    except RateLimitExceededException as e:
        logger.warning(f"Đã vượt quá giới hạn rate limit của GitHub API: {str(e)}")
        
        # Kiểm tra xem còn token nào khác không
        if len(tokens) > 1:
            # Lưu token hiện tại để sau này có thể quay lại
            current_token_index = tokens.index(current_token)
            next_token_index = (current_token_index + 1) % len(tokens)
            next_token = tokens[next_token_index]
            
            logger.info(f"Chuyển sang token tiếp theo ({next_token_index+1}/{len(tokens)})")
            
            # Thử lại với token mới
            try:
                # Lưu trạng thái hiện tại để tiếp tục
                logger.info("Tiếp tục thu thập với token mới...")
                
                # Khởi tạo collector mới với token mới
                collector = GitHubDataCollector(token=next_token)
                
                # Tiếp tục thu thập từ vị trí hiện tại
                collector.collect_and_save_data_with_resume(
                    repos=repos,
                    output_path=output_file,
                    max_commits_per_repo=commits_per_repo,
                    save_chunk_size=1000,
                    start_from_repo_index=current_repo_index,
                    start_from_commit_index=current_commit_index,
                    chunk_start_index=current_chunk_index
                )
                
                logger.info("Đã hoàn thành thu thập dữ liệu với token mới")
                return
                
            except Exception as e2:
                logger.error(f"Lỗi khi thử lại với token mới: {str(e2)}")
        
        # Nếu không thể chuyển token hoặc tất cả token đều hết hạn mức
        if hasattr(e, 'reset_time') and e.reset_time:
            # Tính thời gian cần đợi
            reset_time_str = e.reset_time
            try:
                reset_time = datetime.strptime(reset_time_str, '%Y-%m-%d %H:%M:%S')
                now = datetime.now()
                wait_seconds = max((reset_time - now).total_seconds(), 0)
                wait_time_formatted = time.strftime('%H:%M:%S', time.gmtime(wait_seconds))
                
                logger.info(f"Rate limit sẽ được đặt lại vào: {reset_time_str} (giờ địa phương)")
                logger.info(f"Thời gian cần đợi: {wait_time_formatted} (HH:MM:SS)")
            except (ValueError, TypeError):
                logger.info(f"Rate limit sẽ được đặt lại vào: {reset_time_str} (giờ địa phương)")
                
            logger.info(f"Dữ liệu đã thu thập sẽ được lưu. Bạn có thể tiếp tục thu thập sau thời điểm reset.")
            logger.info(f"Để tiếp tục thu thập, chạy lại script này sau thời điểm reset.")
        else:
            logger.info("Vui lòng thử lại sau khoảng 1 giờ hoặc sử dụng token khác.")
    except Exception as e:
        logger.error(f"Lỗi khi thu thập dữ liệu: {str(e)}")
    finally:
        # Lấy thông tin về repo và commit hiện tại
        # Lưu ý: chỉ cập nhật total_collected nếu chạy thành công và đếm được commit
        state = {
            "total_collected": 0,  # Sẽ được cập nhật trong future version nếu cần
            "current_repo_index": current_repo_index,
            "current_commit_index": current_commit_index,
            "current_chunk_index": current_chunk_index,
            "started_at": state.get("started_at", datetime.now().isoformat()),
            "last_update": datetime.now().isoformat(),
            "last_error": None
        }
        
        # Kiểm tra xem có lỗi rate limit không để lưu vào state
        try:
            if 'e' in locals() and isinstance(e, RateLimitExceededException):
                state["last_error"] = "rate_limit_exceeded"
                state["reset_time"] = getattr(e, 'reset_time', None)
        except Exception:
            pass
        
        # Lưu trạng thái
        try:
            with safe_open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Đã lưu trạng thái thu thập. Để tiếp tục chạy lại script này.")
        except Exception as e:
            logger.error(f"Lỗi khi lưu state file: {str(e)}")
            # Thử lưu vào thư mục hiện tại nếu không lưu được vào output_dir
            fallback_state_file = os.path.join(current_dir, "collection_state.json")
            try:
                with safe_open(fallback_state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                logger.info(f"Đã lưu trạng thái thu thập vào {fallback_state_file}")
            except Exception as e2:
                logger.error(f"Không thể lưu state file: {str(e2)}")

def merge_commit_files(input_dir: str, output_file: str):
    """
    Kết hợp nhiều file commit thành một file duy nhất.
    
    Args:
        input_dir: Thư mục chứa các file commit
        output_file: Đường dẫn file đầu ra
    """
    logger.info(f"Bắt đầu kết hợp các file commit từ {input_dir}")
    
    all_commits = []
    all_repos = set()
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
                    with safe_open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Lấy danh sách commit
                    if 'data' in data:
                        commits = data['data']
                        all_commits.extend(commits)
                        logger.info(f"  -> Đã thêm {len(commits)} commit")
                    
                    # Lấy danh sách repo
                    if 'metadata' in data and 'repositories' in data['metadata']:
                        repos = data['metadata']['repositories']
                        if isinstance(repos, list):
                            all_repos.update(repos)
                    
                except Exception as e:
                    logger.error(f"  -> Lỗi khi đọc file {file}: {str(e)}")
    
    logger.info(f"Đã đọc xong {file_count} file, tổng cộng {len(all_commits)} commit")
    
    if not all_commits:
        logger.warning("Không có commit nào để kết hợp")
        return
    
    # Lưu file kết hợp
    logger.info(f"Đang lưu {len(all_commits)} commit vào {output_file}")
    
    try:
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with safe_open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'total_samples': len(all_commits),
                    'created_at': datetime.now().isoformat(),
                    'repositories': list(all_repos),
                    'source_files': file_count
                },
                'data': all_commits
            }, f, ensure_ascii=False)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"Đã lưu thành công! Kích thước file: {file_size_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Lỗi khi lưu file kết hợp: {str(e)}")

def read_tokens_from_file(token_file: str) -> List[str]:
    """
    Đọc danh sách token từ file.
    
    Args:
        token_file: Đường dẫn đến file chứa token
        
    Returns:
        Danh sách các token hợp lệ
    """
    if not os.path.exists(token_file):
        logger.error(f"Không tìm thấy file token: {token_file}")
        return []
    
    tokens = []
    try:
        with safe_open(token_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Bỏ qua dòng trống hoặc dòng comment
                if line and not line.startswith('#'):
                    tokens.append(line)
        logger.info(f"Đã đọc {len(tokens)} token từ file {token_file}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file token: {str(e)}")
        # Thông báo hướng dẫn
        logger.info("Hãy đảm bảo file token tồn tại và có quyền đọc. Định dạng file mỗi dòng là một token.")
    
    return tokens

def safe_open(file_path, mode='r', encoding='utf-8', **kwargs):
    """
    Mở file một cách an toàn với xử lý lỗi encoding.
    
    Args:
        file_path: Đường dẫn đến file
        mode: Chế độ mở file ('r', 'w', 'a', etc.)
        encoding: Encoding sử dụng (mặc định là utf-8)
        **kwargs: Các tham số khác cho hàm open
        
    Returns:
        File object
    """
    try:
        # Sử dụng đường dẫn tuyệt đối
        abs_path = os.path.abspath(file_path)
        return open(abs_path, mode=mode, encoding=encoding, **kwargs)
    except UnicodeEncodeError as e:
        # Xử lý lỗi Unicode trong đường dẫn
        logger.error(f"Lỗi Unicode khi mở file: {str(e)}")
        logger.info("Thử chuyển đổi đường dẫn sang dạng ngắn gọn...")
        
        try:
            # Thử sử dụng short path name trong Windows
            if os.name == 'nt':  # Windows
                import ctypes
                buffer_size = 500
                buffer = ctypes.create_unicode_buffer(buffer_size)
                get_short_path_name = ctypes.windll.kernel32.GetShortPathNameW
                get_short_path_name(abs_path, buffer, buffer_size)
                short_path = buffer.value
                logger.info(f"Đã chuyển đổi sang đường dẫn ngắn: {short_path}")
                return open(short_path, mode=mode, encoding=encoding, **kwargs)
            else:
                raise e  # Nếu không phải Windows, ném lại ngoại lệ
        except Exception as e2:
            logger.error(f"Không thể mở file với đường dẫn ngắn: {str(e2)}")
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Thu thập và kết hợp commit từ GitHub')
    subparsers = parser.add_subparsers(dest='command', help='Lệnh')
    
    # Lệnh thu thập
    collect_parser = subparsers.add_parser('collect', help='Thu thập commit')
    collect_parser.add_argument('--token', required=False, help='GitHub API token')
    collect_parser.add_argument('--token_file', help='Đường dẫn đến file chứa danh sách token')
    collect_parser.add_argument('--output_dir', default='../data/github_commits', help='Thư mục lưu dữ liệu')
    collect_parser.add_argument('--target', type=int, default=100000, help='Số lượng commit mục tiêu')
    
    # Lệnh kết hợp
    merge_parser = subparsers.add_parser('merge', help='Kết hợp các file commit')
    merge_parser.add_argument('--input_dir', default='../data/github_commits', help='Thư mục chứa các file commit')
    merge_parser.add_argument('--output_file', default='../data/all_commits.json', help='File đầu ra')
    
    args = parser.parse_args()
    
    if args.command == 'collect':
        if not args.token and not args.token_file and not os.environ.get('GITHUB_TOKEN'):
            default_token_file = os.path.join(os.path.dirname(parent_dir), 'data', 'tokens.txt')
            if os.path.exists(default_token_file):
                logger.info(f"Không có token được cung cấp, sử dụng file token mặc định: {default_token_file}")
                collect_100k_commits(token_file=default_token_file, output_dir=args.output_dir, target_commits=args.target)
            else:
                parser.error("Cần cung cấp token thông qua --token, --token_file hoặc biến môi trường GITHUB_TOKEN")
        else:
            collect_100k_commits(github_token=args.token, token_file=args.token_file, output_dir=args.output_dir, target_commits=args.target)
    elif args.command == 'merge':
        merge_commit_files(args.input_dir, args.output_file)
    else:
        parser.print_help()
