"""
Script thu thập commit từ GitHub bằng cách clone repository.
Script này không bị giới hạn bởi GitHub API rate limit.
"""
import os
import sys
import time
import logging
import json
import argparse
from datetime import datetime

# Thêm thư mục gốc vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_collection.git_clone_collector import GitCloneCollector

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, "collect_git_clone.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_repo_list(file_path):
    """Tải danh sách repo từ file."""
    repos = []
    
    # Danh sách các encoding phổ biến để thử
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii', 'utf-16']
    
    for encoding in encodings:
        try:
            repos = []
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        repos.append(line)
            logger.info(f"Đã đọc file repo_list với encoding {encoding}")
            return repos
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Lỗi khi đọc file repo với encoding {encoding}: {str(e)}")
            continue
    
    # Nếu không thể đọc với bất kỳ encoding nào, thử đọc dưới dạng bytes và lọc các ký tự hợp lệ
    try:
        repos = []
        with open(file_path, 'rb') as f:
            for line_bytes in f:
                try:
                    # Loại bỏ các ký tự không hợp lệ và chuyển đổi thành chuỗi
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if line and not line.startswith('#'):
                        repos.append(line)
                except Exception:
                    continue
        logger.info(f"Đã đọc file repo_list bằng phương pháp binary và lọc ký tự")
        return repos
    except Exception as e:
        logger.error(f"Lỗi khi đọc file repo ở chế độ binary: {str(e)}")
    
    # Trường hợp cuối cùng: Tạo danh sách repo mặc định
    logger.warning("Không thể đọc file repo, sử dụng danh sách mặc định")
    return [
        "facebook/react",
        "microsoft/vscode",
        "tensorflow/tensorflow",
        "angular/angular",
        "vuejs/vue"
    ]

def create_default_repo_list(output_file):
    """Tạo file danh sách repo mặc định."""
    repos = [
        "# Danh sách các repository để thu thập commit",
        "# Mỗi dòng một repository (dạng owner/repo)",
        "# Dòng bắt đầu bằng # sẽ bị bỏ qua",
        "",
        "# Frameworks và libraries phổ biến",
        "facebook/react",
        "angular/angular",
        "vuejs/vue",
        "laravel/laravel",
        "django/django",
        "rails/rails",
        "spring-projects/spring-framework",
        "tensorflow/tensorflow",
        "pytorch/pytorch",
        "scikit-learn/scikit-learn",
        "",
        "# Các IDE và công cụ phát triển",
        "microsoft/vscode",
        "JetBrains/intellij-community",
        "atom/atom",
        "neovim/neovim",
        "",
        "# Ngôn ngữ lập trình",
        "golang/go",
        "rust-lang/rust",
        "nodejs/node",
        "python/cpython",
        "ruby/ruby",
        "php/php-src",
        "dotnet/runtime",
        "",
        "# Các dự án lớn khác",
        "kubernetes/kubernetes",
        "docker/docker-ce",
        "apache/spark",
        "elastic/elasticsearch",
        "pandas-dev/pandas",
        "flutter/flutter"
    ]    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(repos))
        logger.info(f"Đã tạo file danh sách repo mặc định với encoding utf-8: {output_file}")
    except Exception as e:
        # Thử lại với encoding khác nếu utf-8 thất bại
        try:
            with open(output_file, 'w', encoding='cp1252') as f:
                f.write('\n'.join(repos))
            logger.info(f"Đã tạo file danh sách repo mặc định với encoding cp1252: {output_file}")
        except Exception as e2:
            logger.error(f"Không thể tạo file danh sách repo: {str(e2)}")

def collect_commits_from_repos(args):
    """Thu thập commit từ các repo."""
    
    # Xác định danh sách repo
    if args.repo_list_file:
        repos = load_repo_list(args.repo_list_file)
        
        # Nếu không thể đọc file chính, thử file đơn giản
        if not repos:
            alt_repo_file = os.path.join(os.path.dirname(args.repo_list_file), "repo_list_simple.txt")
            if os.path.exists(alt_repo_file):
                logger.info(f"Không thể đọc {args.repo_list_file}, thử file thay thế {alt_repo_file}")
                repos = load_repo_list(alt_repo_file)
        
        logger.info(f"Đã tải {len(repos)} repo từ file")
    else:
        # Danh sách repo mặc định
        repos = [
            "facebook/react",
            "microsoft/vscode",
            "tensorflow/tensorflow",
            "angular/angular",
            "vuejs/vue",
            "django/django",
            "laravel/laravel",
            "nodejs/node",
            "pandas-dev/pandas",
            "flutter/flutter"
        ]
        logger.info(f"Sử dụng danh sách {len(repos)} repo mặc định")
    
    # Nếu chỉ định số lượng repo giới hạn
    if args.max_repos > 0 and args.max_repos < len(repos):
        repos = repos[:args.max_repos]
        logger.info(f"Giới hạn số lượng repo: {len(repos)}")
    
    # Xác định đường dẫn output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(parent_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(output_dir, f"git_clone_commits_{timestamp}.json")
    
    # Tạo collector
    collector = GitCloneCollector(temp_dir=args.temp_dir)
    
    # Chia nhỏ danh sách repo để xử lý theo batch
    batch_size = args.batch_size
    total_repos = len(repos)
    
    if args.batch:
        # Xử lý theo batch
        batches = [repos[i:i+batch_size] for i in range(0, total_repos, batch_size)]
        logger.info(f"Chia {total_repos} repo thành {len(batches)} batch, mỗi batch {batch_size} repo")
        
        all_commits = []
        all_repos = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Xử lý batch {i+1}/{len(batches)} với {len(batch)} repo")
            
            # Tạo đường dẫn output cho batch
            batch_output = output_path.replace(".json", f"_batch{i+1}.json")
            
            # Thu thập dữ liệu
            start_time = time.time()
            collector.collect_data_from_repos(
                repos=batch,
                output_path=batch_output,
                max_commits_per_repo=args.max_commits_per_repo,
                shallow_clone=not args.full_clone
            )
            
            # Đọc dữ liệu đã thu thập
            try:
                with open(batch_output, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_commits.extend(data.get('data', []))
                    all_repos.extend(batch)
                    
                elapsed_time = time.time() - start_time
                logger.info(f"Batch {i+1} hoàn thành trong {elapsed_time:.1f} giây, thu được {len(data.get('data', []))} commit")
            except Exception as e:
                logger.error(f"Lỗi khi đọc file batch {batch_output}: {str(e)}")
        
        # Lưu tất cả dữ liệu vào file chính
        metadata = {
            'total_samples': len(all_commits),
            'created_at': datetime.now().isoformat(),
            'repositories': all_repos,
            'collection_method': 'git_clone'
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': metadata,
                'data': all_commits
            }, f, ensure_ascii=False)
        
        logger.info(f"Đã kết hợp tất cả {len(all_commits)} commit từ {len(batches)} batch vào {output_path}")
        
    else:
        # Xử lý tất cả repo cùng lúc
        logger.info(f"Bắt đầu thu thập dữ liệu từ {total_repos} repo")
        collector.collect_data_from_repos(
            repos=repos,
            output_path=output_path,
            max_commits_per_repo=args.max_commits_per_repo,
            shallow_clone=not args.full_clone
        )
    
    logger.info(f"Thu thập dữ liệu hoàn tất. Dữ liệu được lưu vào {output_path}")

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Thu thập commit bằng cách clone repository')
    
    # Tạo các subparser cho các lệnh
    subparsers = parser.add_subparsers(dest='command', help='Lệnh')
    
    # Lệnh tạo file danh sách repo
    create_parser = subparsers.add_parser('create-repo-list', help='Tạo file danh sách repo mặc định')
    create_parser.add_argument('--output', type=str, default='data/repo_list.txt',
                             help='Đường dẫn đến file đầu ra (mặc định: data/repo_list.txt)')
    
    # Lệnh thu thập
    collect_parser = subparsers.add_parser('collect', help='Thu thập commit từ repository')
    collect_parser.add_argument('--repo_list_file', type=str, 
                              help='Đường dẫn tới file chứa danh sách các repository')
    collect_parser.add_argument('--output_file', type=str,
                              help='Đường dẫn file output (mặc định: data/git_clone_commits_<timestamp>.json)')
    collect_parser.add_argument('--temp_dir', type=str, default='temp_repos',
                              help='Thư mục tạm để clone repository (mặc định: temp_repos)')
    collect_parser.add_argument('--max_commits_per_repo', type=int, default=1000,
                              help='Số lượng commit tối đa từ mỗi repo (mặc định: 1000)')
    collect_parser.add_argument('--full_clone', action='store_true',
                              help='Clone toàn bộ repository thay vì shallow clone')
    collect_parser.add_argument('--batch', action='store_true',
                              help='Xử lý theo batch')
    collect_parser.add_argument('--batch_size', type=int, default=5,
                              help='Số lượng repo xử lý trong mỗi batch (mặc định: 5)')
    collect_parser.add_argument('--max_repos', type=int, default=0,
                              help='Số lượng repo tối đa cần xử lý (0 = tất cả)')
    
    return parser.parse_args()

def run_interactive():
    """Chạy chế độ tương tác."""
    print("\n=== Thu thập commit từ GitHub bằng Git Clone ===\n")
      # Xác định thư mục data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Kiểm tra file danh sách repo
    repo_file = os.path.join(data_dir, "repo_list.txt")
    if not os.path.exists(repo_file):
        print("Không tìm thấy file danh sách repository.")
        create = input("Bạn có muốn tạo file danh sách repository mặc định không? (y/n): ").strip().lower()
        if create == 'y':
            create_default_repo_list(repo_file)
            print(f"Đã tạo file {repo_file}. Vui lòng chỉnh sửa file này trước khi tiếp tục.")
            return
    
    # Thử file đơn giản nếu file gốc có vấn đề
    alt_repo_file = os.path.join(data_dir, "repo_list_simple.txt")
    
    # Đọc danh sách repo
    repos = load_repo_list(repo_file)
    if not repos and os.path.exists(alt_repo_file):
        print(f"Không thể đọc {repo_file}, thử file thay thế {alt_repo_file}")
        repos = load_repo_list(alt_repo_file)
    
    if not repos:
        print(f"Không tìm thấy repository nào trong các file danh sách")
        # Hỏi người dùng có muốn sử dụng danh sách mặc định không
        use_default = input("Sử dụng danh sách mặc định? (y/n): ").strip().lower()
        if use_default != 'y':
            return
        repos = [
            "facebook/react",
            "microsoft/vscode",
            "tensorflow/tensorflow",
            "angular/angular",
            "vuejs/vue"
        ]
    
    print(f"Đã tìm thấy {len(repos)} repositories trong file {repo_file}.")
    
    # Hỏi số lượng repo cần thu thập
    try:
        max_repos = int(input("Số lượng repository tối đa cần thu thập (0 = tất cả, Enter = tất cả): ") or "0")
    except ValueError:
        print("Giá trị không hợp lệ, sử dụng tất cả repositories")
        max_repos = 0
    
    if max_repos > 0 and max_repos < len(repos):
        repos = repos[:max_repos]
        print(f"Sẽ thu thập từ {len(repos)} repositories đầu tiên")
    
    # Hỏi số lượng commit tối đa từ mỗi repo
    try:
        max_commits = int(input("Số lượng commit tối đa từ mỗi repository (mặc định 1000): ") or "1000")
    except ValueError:
        print("Giá trị không hợp lệ, sử dụng mặc định 1000")
        max_commits = 1000
    
    # Hỏi có xử lý theo batch không
    use_batch = input("Xử lý theo batch? (y/n, mặc định y): ").strip().lower() != 'n'
    
    batch_size = 5
    if use_batch:
        try:
            batch_size = int(input("Số lượng repository trong mỗi batch (mặc định 5): ") or "5")
        except ValueError:
            print("Giá trị không hợp lệ, sử dụng mặc định 5")
            batch_size = 5
    
    # Hỏi có clone đầy đủ không
    full_clone = input("Clone đầy đủ repository? (y/n, mặc định n): ").strip().lower() == 'y'
    
    # Xác định thư mục tạm
    temp_dir = os.path.join(data_dir, "temp_repos")
    
    # Xác định file đầu ra
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(data_dir, f"git_clone_commits_{timestamp}.json")
    
    # Hiển thị thông tin
    print("\n=== Thông tin thu thập ===")
    print(f"Số lượng repository: {len(repos)}")
    print(f"Số commit tối đa mỗi repo: {max_commits}")
    print(f"Phương thức clone: {'Đầy đủ' if full_clone else 'Nông (shallow)'}")
    print(f"Xử lý theo batch: {'Có, mỗi batch {0} repo'.format(batch_size) if use_batch else 'Không'}")
    print(f"Thư mục tạm: {temp_dir}")
    print(f"File đầu ra: {output_file}")
    
    # Xác nhận
    confirm = input("\nBạn có chắc chắn muốn bắt đầu thu thập? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Đã hủy thu thập dữ liệu")
        return
    
    # Tạo đối tượng args giả lập
    class Args:
        pass
    
    args = Args()
    args.repo_list_file = repo_file
    args.output_file = output_file
    args.temp_dir = temp_dir
    args.max_commits_per_repo = max_commits
    args.full_clone = full_clone
    args.batch = use_batch
    args.batch_size = batch_size
    args.max_repos = max_repos
    
    # Thu thập dữ liệu
    try:
        print("\nBắt đầu thu thập dữ liệu...")
        collect_commits_from_repos(args)
    except KeyboardInterrupt:
        print("\nThu thập bị ngắt bởi người dùng")
    except Exception as e:
        print(f"\nLỗi khi thu thập dữ liệu: {str(e)}")

if __name__ == "__main__":
    args = parse_args()
    
    try:
        if args.command == 'create-repo-list':
            create_default_repo_list(args.output)
        elif args.command == 'collect':
            collect_commits_from_repos(args)
        else:
            # Nếu không có lệnh cụ thể, chạy chế độ tương tác
            run_interactive()
    except KeyboardInterrupt:
        logger.info("Thu thập bị ngắt bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
