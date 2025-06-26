"""
Module thu thập dữ liệu commit từ GitHub API.
"""
import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("github_collector.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimitExceededException(Exception):
    """Ngoại lệ khi vượt quá giới hạn rate limit của GitHub API."""
    
    def __init__(self, message, reset_time=None):
        self.message = message
        self.reset_time = reset_time
        super().__init__(self.message)

class GitHubDataCollector:
    """Class thu thập dữ liệu commit từ GitHub API."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Khởi tạo collector với GitHub token.
        
        Args:
            token: GitHub API token (nếu không cung cấp, sẽ sử dụng giá trị từ biến môi trường GITHUB_TOKEN)
        """
        self.token = token or os.environ.get('GITHUB_TOKEN')
        if not self.token:
            logger.warning("GitHub token không được cung cấp. API rate limit sẽ bị hạn chế.")
        
        self.headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
        
        self.base_url = 'https://api.github.com'
    
    def get_commit_history(self, repo: str, branch: str = 'main', max_pages: int = 10, per_page: int = 100) -> List[Dict]:
        """
        Lấy lịch sử commit từ repository.
        
        Args:
            repo: Tên repository (format: 'owner/repo')
            branch: Tên nhánh
            max_pages: Số trang tối đa cần lấy
            per_page: Số commit mỗi trang
            
        Returns:
            List các commit
            
        Raises:
            RateLimitExceededException: Nếu vượt quá giới hạn rate limit
        """
        commits = []
        page = 1
        
        while page <= max_pages:
            # Kiểm tra rate limit trước khi gửi request lớn
            self._check_rate_limit_before_request()
            
            url = f"{self.base_url}/repos/{repo}/commits"
            params = {
                'sha': branch,
                'per_page': per_page,
                'page': page
            }
            
            logger.info(f"Đang lấy commit từ {repo}, nhánh {branch}, trang {page}")
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                page_commits = response.json()
                if not page_commits:
                    break  # Không còn commit
                
                commits.extend(page_commits)
                page += 1
                
                # Đợi để tránh vượt quá rate limit
                time.sleep(0.5)
            elif response.status_code == 403 and 'rate limit exceeded' in response.text:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - time.time(), 0) + 1
                logger.warning(f"Rate limit exceeded. Đợi {wait_time:.0f} giây.")
                time.sleep(wait_time)
            else:
                logger.error(f"Lỗi khi lấy commit: {response.status_code} - {response.text}")
                break
        
        return commits
    
    def get_commit_details(self, repo: str, commit_sha: str) -> Dict:
        """
        Lấy chi tiết một commit cụ thể.
        
        Args:
            repo: Tên repository (format: 'owner/repo')
            commit_sha: SHA của commit
            
        Returns:
            Chi tiết commit
            
        Raises:
            RateLimitExceededException: Nếu vượt quá giới hạn rate limit
        """
        # Kiểm tra rate limit trước khi gửi request
        self._check_rate_limit_before_request()
        
        url = f"{self.base_url}/repos/{repo}/commits/{commit_sha}"
        
        logger.info(f"Đang lấy chi tiết commit {commit_sha} từ {repo}")
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            reset_time_str = datetime.fromtimestamp(reset_time).strftime('%Y-%m-%d %H:%M:%S')
            raise RateLimitExceededException(
                f"Rate limit exceeded khi lấy chi tiết commit.",
                reset_time_str
            )
        else:
            logger.error(f"Lỗi khi lấy chi tiết commit: {response.status_code} - {response.text}")
            return {}
    
    def extract_commit_data(self, commit: Dict) -> Dict:
        """
        Trích xuất thông tin quan trọng từ dữ liệu commit.
        
        Args:
            commit: Dữ liệu commit từ GitHub API
            
        Returns:
            Dữ liệu đã được xử lý
        """
        commit_data = {
            'text': commit.get('commit', {}).get('message', ''),
            'metadata': {
                'commit_id': commit.get('sha', ''),
                'author': commit.get('commit', {}).get('author', {}).get('name', 'unknown'),
                'author_email': commit.get('commit', {}).get('author', {}).get('email', ''),
                'timestamp': commit.get('commit', {}).get('author', {}).get('date', ''),
                'files_changed': len(commit.get('files', [])),
                'additions': sum(f.get('additions', 0) for f in commit.get('files', [])),
                'deletions': sum(f.get('deletions', 0) for f in commit.get('files', [])),
                'total_changes': sum((f.get('additions', 0) + f.get('deletions', 0)) for f in commit.get('files', [])),
                'is_merge': len(commit.get('parents', [])) > 1,
                'modified_files': [f.get('filename') for f in commit.get('files', [])],
                'file_types': self._extract_file_types(commit.get('files', [])),
                'modified_directories': self._extract_directories(commit.get('files', [])),
            }
        }
        
        return commit_data
    
    def _extract_file_types(self, files: List[Dict]) -> Dict[str, int]:
        """Trích xuất số lượng file theo loại file."""
        file_types = {}
        for file in files:
            filename = file.get('filename', '')
            ext = os.path.splitext(filename)[1].lower()
            if ext:
                file_types[ext] = file_types.get(ext, 0) + 1
            else:
                file_types['no_extension'] = file_types.get('no_extension', 0) + 1
        return file_types
    
    def _extract_directories(self, files: List[Dict]) -> Dict[str, int]:
        """Trích xuất số lượng file theo thư mục."""
        directories = {}
        for file in files:
            filename = file.get('filename', '')
            directory = os.path.dirname(filename)
            if directory:
                directories[directory] = directories.get(directory, 0) + 1
            else:
                directories['root'] = directories.get('root', 0) + 1
        return directories
    
    def handle_rate_limit(self, response: requests.Response) -> float:
        """
        Xử lý khi gặp rate limit từ GitHub API.
        
        Args:
            response: Response từ GitHub API
            
        Returns:
            Thời gian đã đợi (giây)
        """
        if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = max(reset_time - time.time(), 0) + 1
            logger.warning(f"Rate limit exceeded. Đợi {wait_time:.0f} giây.")
            time.sleep(wait_time)
            return wait_time
        return 0
    
    def check_rate_limit(self) -> Dict[str, Any]:
        """
        Kiểm tra rate limit hiện tại của GitHub API.
        
        Returns:
            Dict chứa thông tin về limit, remaining và reset time
        
        Raises:
            requests.RequestException: Nếu có lỗi khi gọi API
        """
        url = "https://api.github.com/rate_limit"
        headers = {"Authorization": f"token {self.token}"}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return {
                'limit': data['resources']['core']['limit'],
                'remaining': data['resources']['core']['remaining'],
                'reset': data['resources']['core']['reset'],
                'reset_time': datetime.fromtimestamp(data['resources']['core']['reset']).strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            # Nếu không thể kiểm tra rate limit, giả định là còn đủ
            current_time = time.time()
            return {
                'limit': 5000, 
                'remaining': 1000, 
                'reset': current_time + 3600,
                'reset_time': datetime.fromtimestamp(current_time + 3600).strftime('%Y-%m-%d %H:%M:%S')
            }
            
    def _check_rate_limit_before_request(self):
        """
        Kiểm tra rate limit trước khi gửi request và ném ngoại lệ nếu gần hết.
        
        Raises:
            RateLimitExceededException: Nếu số request còn lại quá thấp
        """
        rate_limit = self.check_rate_limit()
        if rate_limit['remaining'] < 10:  # Giữ lại một số request dự phòng
            raise RateLimitExceededException(
                f"Rate limit còn lại quá thấp: {rate_limit['remaining']}/{rate_limit['limit']}",
                rate_limit['reset_time']
            )
    
    def _save_intermediate_data(self, data: List[Dict], output_path: str, repos: List[str], 
                               is_final: bool = False, chunk_index: int = 0) -> None:
        """
        Lưu dữ liệu trung gian để có thể tiếp tục thu thập sau khi gián đoạn.
        
        Args:
            data: Dữ liệu commit đã thu thập
            output_path: Đường dẫn cơ sở để lưu dữ liệu
            repos: Danh sách repositories đã thu thập
            is_final: Có phải là dữ liệu cuối cùng không
            chunk_index: Chỉ số của phần dữ liệu đang lưu
        """
        # Tạo tên file dựa trên chỉ số chunk
        base_name, ext = os.path.splitext(output_path)
        if is_final:
            save_path = output_path
        else:
            save_path = f"{base_name}_part{chunk_index}{ext}"
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Lưu dữ liệu
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_samples': len(data),
                    'created_at': datetime.now().isoformat(),
                    'repositories': repos,
                    'is_complete': is_final,
                    'chunk_index': chunk_index
                },
                'data': data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Đã lưu {len(data)} commit vào {save_path}")
    
    def collect_and_save_data_with_resume(
        self, 
        repos: List[str], 
        output_path: str, 
        max_commits_per_repo: int = 1000,
        save_chunk_size: int = 1000,
        start_from_repo_index: int = 0,
        start_from_commit_index: int = 0,
        chunk_start_index: int = 0
    ) -> None:
        """
        Thu thập và lưu dữ liệu từ nhiều repositories với khả năng tiếp tục sau khi gián đoạn.
        
        Args:
            repos: Danh sách repositories (format: 'owner/repo')
            output_path: Đường dẫn để lưu dữ liệu
            max_commits_per_repo: Số lượng commit tối đa cần lấy từ mỗi repo
            save_chunk_size: Số lượng commit để lưu mỗi đợt
            start_from_repo_index: Bắt đầu từ chỉ số repo nào trong danh sách
            start_from_commit_index: Bắt đầu từ chỉ số commit nào trong repo hiện tại
            chunk_start_index: Chỉ số bắt đầu cho các file chunk
            
        Raises:
            RateLimitExceededException: Nếu vượt quá giới hạn rate limit
        """
        all_commit_data = []
        current_chunk = chunk_start_index
        
        for repo_idx, repo in enumerate(repos[start_from_repo_index:], start=start_from_repo_index):
            try:
                logger.info(f"Đang thu thập dữ liệu từ {repo} ({repo_idx+1}/{len(repos)})")
                
                # Kiểm tra rate limit trước khi lấy commit history
                self._check_rate_limit_before_request()
                
                # Lấy commit history
                commits = self.get_commit_history(
                    repo, 
                    max_pages=(max_commits_per_repo // 100) + 1,
                    per_page=100
                )
                
                # Giới hạn số lượng commit
                commits = commits[:max_commits_per_repo]
                
                # Xác định điểm bắt đầu (chỉ áp dụng cho repo đầu tiên sau khi resume)
                start_commit = start_from_commit_index if repo_idx == start_from_repo_index else 0
                
                # Lấy chi tiết từng commit
                for commit_idx, commit in enumerate(commits[start_commit:], start=start_commit):
                    commit_sha = commit.get('sha')
                    if commit_sha:
                        logger.info(f"Đang lấy commit {commit_idx+1}/{len(commits)} từ {repo}")
                        
                        # Kiểm tra rate limit trước khi lấy chi tiết commit
                        self._check_rate_limit_before_request()
                        
                        commit_detail = self.get_commit_details(repo, commit_sha)
                        
                        if commit_detail:
                            commit_data = self.extract_commit_data(commit_detail)
                            commit_data['metadata']['repository'] = repo
                            all_commit_data.append(commit_data)
                        
                        # Lưu dữ liệu theo từng đợt
                        if len(all_commit_data) >= save_chunk_size:
                            self._save_intermediate_data(
                                all_commit_data, output_path, repos[:repo_idx+1], 
                                is_final=False, chunk_index=current_chunk
                            )
                            current_chunk += 1
                            all_commit_data = []
                
                logger.info(f"Đã thu thập {len(commits)-start_commit} commit từ {repo}")
            
            except RateLimitExceededException as e:
                logger.warning(f"Đã vượt quá giới hạn rate limit của GitHub API: {str(e)}")
                # Lưu trạng thái hiện tại để có thể resume
                if all_commit_data:
                    self._save_intermediate_data(
                        all_commit_data, output_path, repos[:repo_idx+1], 
                        is_final=False, chunk_index=current_chunk
                    )
                
                logger.info(f"Có thể tiếp tục thu thập từ repo_index={repo_idx}, commit_index={commit_idx if 'commit_idx' in locals() else 0}, chunk_index={current_chunk+1}")
                # Propagate ngoại lệ để caller xử lý
                raise
                
            except Exception as e:
                logger.error(f"Lỗi khi thu thập dữ liệu từ {repo}: {str(e)}")
                # Lưu trạng thái hiện tại để có thể resume
                if all_commit_data:
                    self._save_intermediate_data(
                        all_commit_data, output_path, repos[:repo_idx+1], 
                        is_final=False, chunk_index=current_chunk
                    )
                
                logger.info(f"Có thể tiếp tục thu thập từ repo_index={repo_idx}, commit_index={commit_idx if 'commit_idx' in locals() else 0}, chunk_index={current_chunk+1}")
                return
        
        # Lưu dữ liệu còn lại
        if all_commit_data:
            self._save_intermediate_data(all_commit_data, output_path, repos, is_final=True)
        
        logger.info(f"Đã hoàn thành thu thập dữ liệu từ tất cả {len(repos)} repositories")
    
    def collect_and_save_data(self, repos: List[str], output_path: str, max_commits_per_repo: int = 1000) -> None:
        """
        Thu thập và lưu dữ liệu từ nhiều repositories.
        
        Args:
            repos: Danh sách repositories (format: 'owner/repo')
            output_path: Đường dẫn để lưu dữ liệu
            max_commits_per_repo: Số lượng commit tối đa cần lấy từ mỗi repo
        """
        self.collect_and_save_data_with_resume(repos, output_path, max_commits_per_repo)
