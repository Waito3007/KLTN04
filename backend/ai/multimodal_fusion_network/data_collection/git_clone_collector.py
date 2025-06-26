"""
Module thu thập dữ liệu commit bằng cách clone repository và phân tích lịch sử commit.
Phương pháp này không bị giới hạn bởi GitHub API rate limit.
"""
import os
import subprocess
import json
import time
import logging
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitCloneCollector:
    """Thu thập dữ liệu commit bằng cách clone repository và phân tích lịch sử commit."""
    
    def __init__(self, temp_dir: str = "temp_repos"):
        """
        Khởi tạo GitCloneCollector.
        
        Args:
            temp_dir: Thư mục tạm để clone repository
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    def collect_data_from_repos(self, 
                               repos: List[str], 
                               output_path: str, 
                               max_commits_per_repo: int = 1000,
                               shallow_clone: bool = True) -> None:
        """
        Thu thập dữ liệu từ danh sách repositories.
        
        Args:
            repos: Danh sách repositories (dạng "owner/repo")
            output_path: Đường dẫn để lưu dữ liệu
            max_commits_per_repo: Số lượng commit tối đa cho mỗi repo
            shallow_clone: Clone nông (chỉ lấy một số commit gần nhất) để tiết kiệm thời gian và dung lượng
        """
        all_commit_data = []
        total_collected = 0
        
        for repo_idx, repo in enumerate(repos):
            logger.info(f"[{repo_idx+1}/{len(repos)}] Đang thu thập dữ liệu từ {repo}")
            
            try:
                # Clone repository
                repo_path = self._clone_repository(repo, shallow_clone, max_commits_per_repo)
                if not repo_path:
                    logger.error(f"Không thể clone repository {repo}. Bỏ qua.")
                    continue
                
                # Lấy lịch sử commit
                commits = self._get_commit_history(repo_path, max_commits_per_repo)
                logger.info(f"Đã tìm thấy {len(commits)} commit trong {repo}")
                
                # Thu thập chi tiết từng commit
                for commit_idx, commit_hash in enumerate(commits):
                    try:
                        # Hiển thị tiến độ
                        print(f"\rĐang xử lý: Repo {repo_idx+1}/{len(repos)}, "
                              f"Commit {commit_idx+1}/{len(commits)} - Tổng: {total_collected}", end="")
                        
                        # Lấy chi tiết commit
                        commit_data = self._get_commit_details(repo_path, commit_hash, repo)
                        if commit_data:
                            all_commit_data.append(commit_data)
                            total_collected += 1
                            
                            # Lưu theo chu kỳ để tránh mất dữ liệu
                            if total_collected % 1000 == 0:
                                self._save_data(all_commit_data, output_path, repos, False)
                                logger.info(f"Đã lưu trung gian {total_collected} commit.")
                    
                    except Exception as e:
                        logger.error(f"Lỗi khi xử lý commit {commit_hash} từ {repo}: {str(e)}")
                
                # Xóa thư mục repository sau khi xử lý xong
                self._cleanup_repo(repo_path)
                logger.info(f"Đã xóa thư mục tạm {repo_path}")
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý repository {repo}: {str(e)}")
        
        # Lưu toàn bộ dữ liệu
        self._save_data(all_commit_data, output_path, repos, True)
        logger.info(f"Hoàn thành thu thập dữ liệu. Đã thu thập {total_collected} commit.")
    
    def _clone_repository(self, repo: str, shallow: bool = True, depth: int = 1000) -> Optional[str]:
        """
        Clone repository về máy.
        
        Args:
            repo: Tên repository (dạng "owner/repo")
            shallow: Clone nông
            depth: Số lượng commit tối đa khi clone nông
            
        Returns:
            Đường dẫn đến repository đã clone hoặc None nếu thất bại        """
        try:
            # Tạo thư mục tạm cho repository
            repo_dir = os.path.join(self.temp_dir, repo.replace('/', '_'))
            
            # Xử lý nếu thư mục đã tồn tại
            if os.path.exists(repo_dir):
                logger.info(f"Thư mục {repo_dir} đã tồn tại, cố gắng xóa...")
                try:
                    # Thử thay đổi quyền truy cập các file
                    for root, dirs, files in os.walk(repo_dir, topdown=False):
                        for name in files:
                            try:
                                file_path = os.path.join(root, name)
                                os.chmod(file_path, 0o777)  # Thay đổi quyền truy cập
                            except:
                                pass
                    
                    # Xóa thư mục
                    shutil.rmtree(repo_dir)
                    logger.info(f"Đã xóa thành công thư mục {repo_dir}")
                except Exception as rm_error:
                    # Nếu không thể xóa, tạo một thư mục mới với timestamp
                    logger.warning(f"Không thể xóa thư mục {repo_dir}: {str(rm_error)}")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    repo_dir = os.path.join(self.temp_dir, f"{repo.replace('/', '_')}_{timestamp}")
                    logger.info(f"Sử dụng thư mục mới: {repo_dir}")
            
            # Tạo lệnh clone
            clone_url = f"https://github.com/{repo}.git"
            clone_command = ["git", "clone"]
            
            if shallow:
                clone_command.extend(["--depth", str(depth)])
            
            clone_command.extend([clone_url, repo_dir])
              # Thực hiện clone
            logger.info(f"Đang clone {clone_url} vào {repo_dir}")
            
            # Thêm thời gian chờ để đảm bảo các file không còn bị lock
            time.sleep(1)
            
            try:
                process = subprocess.run(
                    clone_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if process.returncode != 0:
                    logger.error(f"Lỗi khi clone repository {repo}: {process.stderr}")
                    return None
                
                logger.info(f"Đã clone thành công {repo}")
                return repo_dir
            except Exception as clone_error:
                # Thử phương pháp clone thay thế
                logger.warning(f"Lỗi khi clone {repo} với phương pháp thông thường: {str(clone_error)}")
                logger.info(f"Thử phương pháp clone thay thế cho {repo}...")
                
                # Tạo thư mục trước
                os.makedirs(repo_dir, exist_ok=True)
                
                # Sử dụng subprocess.Popen thay vì subprocess.run
                try:
                    process = subprocess.Popen(
                        clone_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=None
                    )
                    stdout, stderr = process.communicate(timeout=600)  # timeout 10 phút
                    
                    if process.returncode != 0:
                        logger.error(f"Lỗi khi clone repository (phương pháp thay thế) {repo}: {stderr}")
                        return None
                    
                    logger.info(f"Đã clone thành công {repo} (phương pháp thay thế)")
                    return repo_dir
                except Exception as alt_error:
                    logger.error(f"Lỗi khi clone repository với phương pháp thay thế {repo}: {str(alt_error)}")
                    return None
            
        except Exception as e:
            logger.error(f"Lỗi khi clone repository {repo}: {str(e)}")
            return None
    
    def _get_commit_history(self, repo_path: str, max_commits: int = 1000) -> List[str]:
        """
        Lấy lịch sử commit từ repository đã clone.
        
        Args:
            repo_path: Đường dẫn đến repository
            max_commits: Số lượng commit tối đa cần lấy
            
        Returns:
            Danh sách hash của các commit
        """
        try:
            # Tạo lệnh git log
            log_command = [
                "git", "log", 
                f"-{max_commits}", 
                "--pretty=format:%H"
            ]
            
            # Thực hiện lệnh
            process = subprocess.run(
                log_command,
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"Lỗi khi lấy lịch sử commit: {process.stderr}")
                return []
            
            # Phân tích kết quả
            commit_hashes = process.stdout.strip().split('\n')
            return commit_hashes
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy lịch sử commit: {str(e)}")
            return []
    
    def _get_commit_details(self, repo_path: str, commit_hash: str, repo_name: str) -> Optional[Dict]:
        """
        Lấy chi tiết của một commit.
        
        Args:
            repo_path: Đường dẫn đến repository
            commit_hash: Hash của commit
            repo_name: Tên repository (dạng "owner/repo")
            
        Returns:
            Dữ liệu commit đã được xử lý
        """
        try:
            # Lấy thông tin commit message và author
            show_command = [
                "git", "show", 
                commit_hash, 
                "--pretty=format:%an%n%ae%n%at%n%s%n%b"
            ]
            
            show_process = subprocess.run(
                show_command,
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if show_process.returncode != 0:
                logger.error(f"Lỗi khi lấy thông tin commit {commit_hash}: {show_process.stderr}")
                return None
            
            # Phân tích kết quả
            lines = show_process.stdout.split('\n')
            author_name = lines[0]
            author_email = lines[1]
            author_time = int(lines[2])  # Unix timestamp
            subject = lines[3]
            body = '\n'.join(lines[4:])
            
            commit_message = subject
            if body.strip():
                commit_message += '\n\n' + body
            
            # Lấy danh sách file thay đổi
            diff_command = [
                "git", "diff-tree", 
                "--no-commit-id", 
                "--name-status", 
                "-r",
                commit_hash
            ]
            
            diff_process = subprocess.run(
                diff_command,
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if diff_process.returncode != 0:
                logger.error(f"Lỗi khi lấy danh sách file thay đổi: {diff_process.stderr}")
                return None
            
            # Phân tích danh sách file
            changed_files = []
            for line in diff_process.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 2:
                    status = parts[0]
                    filename = parts[1]
                    changed_files.append({
                        'status': status,
                        'filename': filename
                    })
            
            # Lấy số dòng thêm/xóa
            stat_command = [
                "git", "diff", 
                "--numstat",
                f"{commit_hash}^..{commit_hash}"
            ]
            
            try:
                stat_process = subprocess.run(
                    stat_command,
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                additions = 0
                deletions = 0
                
                if stat_process.returncode == 0:
                    for line in stat_process.stdout.strip().split('\n'):
                        if not line:
                            continue
                            
                        parts = line.split('\t')
                        if len(parts) >= 3 and parts[0] != '-' and parts[1] != '-':
                            try:
                                additions += int(parts[0])
                                deletions += int(parts[1])
                            except ValueError:
                                pass
            except Exception:
                # Nếu có lỗi khi lấy stat, vẫn tiếp tục với giá trị mặc định
                additions = 0
                deletions = 0
            
            # Tạo dữ liệu commit
            commit_data = {
                'text': commit_message,
                'metadata': {
                    'commit_id': commit_hash,
                    'author': author_name,
                    'author_email': author_email,
                    'timestamp': datetime.fromtimestamp(author_time).isoformat(),
                    'repository': repo_name,
                    'files_changed': len(changed_files),
                    'additions': additions,
                    'deletions': deletions,
                    'total_changes': additions + deletions,
                    'modified_files': [f['filename'] for f in changed_files],
                    'file_types': self._extract_file_types([f['filename'] for f in changed_files]),
                    'modified_directories': self._extract_directories([f['filename'] for f in changed_files]),
                }
            }
            
            # Kiểm tra nếu là merge commit
            merge_check_command = ["git", "rev-list", "--parents", "-n", "1", commit_hash]
            merge_process = subprocess.run(
                merge_check_command,
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if merge_process.returncode == 0:
                parents = merge_process.stdout.strip().split()
                if len(parents) > 2:  # Commit hash + 2 or more parents = merge commit
                    commit_data['metadata']['is_merge'] = True
                else:
                    commit_data['metadata']['is_merge'] = False
            
            return commit_data
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy chi tiết commit {commit_hash}: {str(e)}")
            return None
    
    def _extract_file_types(self, filenames: List[str]) -> Dict[str, int]:
        """Trích xuất số lượng file theo loại file."""
        file_types = {}
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext:
                file_types[ext] = file_types.get(ext, 0) + 1
            else:
                file_types['no_extension'] = file_types.get('no_extension', 0) + 1
        return file_types
    
    def _extract_directories(self, filenames: List[str]) -> Dict[str, int]:
        """Trích xuất số lượng file theo thư mục."""
        directories = {}
        for filename in filenames:
            directory = os.path.dirname(filename)
            if directory:
                directories[directory] = directories.get(directory, 0) + 1
            else:
                directories['root'] = directories.get('root', 0) + 1
        return directories
    def _cleanup_repo(self, repo_path: str) -> None:
        """Xóa thư mục repository sau khi xử lý xong."""
        if not os.path.exists(repo_path):
            return
            
        try:
            # Thử thay đổi quyền truy cập các file trước khi xóa
            for root, dirs, files in os.walk(repo_path, topdown=False):
                for name in files:
                    try:
                        file_path = os.path.join(root, name)
                        os.chmod(file_path, 0o777)  # Thay đổi quyền truy cập
                    except:
                        pass
            
            # Thêm thời gian chờ để đảm bảo các file không còn bị lock
            time.sleep(1)
            
            # Xóa thư mục
            shutil.rmtree(repo_path)
            logger.info(f"Đã xóa thành công thư mục {repo_path}")
        except Exception as e:
            logger.error(f"Lỗi khi xóa thư mục {repo_path}: {str(e)}")
            # Thử phương pháp khác nếu shutil.rmtree thất bại
            try:
                # Sử dụng lệnh hệ thống để xóa
                if os.name == 'nt':  # Windows
                    os.system(f'rd /s /q "{repo_path}"')
                else:  # Linux/Mac
                    os.system(f'rm -rf "{repo_path}"')
                logger.info(f"Đã xóa thành công thư mục {repo_path} (phương pháp thay thế)")
            except Exception as alt_e:
                logger.error(f"Không thể xóa thư mục {repo_path} với phương pháp thay thế: {str(alt_e)}")
    
    def _save_data(self, data: List[Dict], output_path: str, repos: List[str], is_final: bool = True) -> None:
        """Lưu dữ liệu vào file JSON."""
        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Tạo metadata
            metadata = {
                'total_samples': len(data),
                'created_at': datetime.now().isoformat(),
                'repositories': repos,
                'collection_method': 'git_clone'
            }
            
            # Lưu dữ liệu
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': metadata,
                    'data': data
                }, f, ensure_ascii=False)
                
            if is_final:
                logger.info(f"Đã lưu {len(data)} commit vào {output_path}")
                
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu: {str(e)}")


if __name__ == "__main__":
    # Ví dụ sử dụng
    collector = GitCloneCollector(temp_dir="temp_repos")
    
    # Danh sách repo muốn thu thập
    repos = [
        "facebook/react",
        "microsoft/vscode",
        "tensorflow/tensorflow"
    ]
    
    # Đường dẫn lưu dữ liệu
    output_path = "data/git_clone_commits.json"
    
    # Thu thập dữ liệu
    collector.collect_data_from_repos(
        repos=repos,
        output_path=output_path,
        max_commits_per_repo=500,
        shallow_clone=True  # True để clone nông, tiết kiệm thời gian và dung lượng
    )
