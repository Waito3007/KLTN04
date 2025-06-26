"""
Module thu thập dữ liệu commit từ GitHub API với khả năng xử lý song song.
"""
import os
import json
import time
import logging
import requests
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParallelGitHubCollector:
    """Class thu thập dữ liệu commit từ GitHub API với khả năng xử lý song song."""
    
    def __init__(self, token: Optional[str] = None, token_list: Optional[List[str]] = None):
        """
        Khởi tạo collector với GitHub token hoặc danh sách token.
        
        Args:
            token: GitHub API token chính
            token_list: Danh sách GitHub API token để sử dụng luân phiên (tăng rate limit)
        """
        self.token = token or os.environ.get('GITHUB_TOKEN')
        self.token_list = token_list or []
        
        # Thêm token chính vào danh sách nếu không có trong token_list
        if self.token and self.token not in self.token_list:
            self.token_list.append(self.token)
        
        if not self.token_list:
            logger.warning("GitHub token không được cung cấp. API rate limit sẽ bị hạn chế.")
        else:
            logger.info(f"Đã khởi tạo collector với {len(self.token_list)} token")
        
        self.current_token_index = 0
        self.base_url = 'https://api.github.com'
        
        # Thống kê API calls để tối ưu sử dụng token
        self.api_calls = {token: 0 for token in self.token_list}
        self.rate_limit_reset = {token: 0 for token in self.token_list}
        self.rate_limit_remaining = {token: 0 for token in self.token_list}
    
    def _get_headers(self, token_index: Optional[int] = None) -> Dict[str, str]:
        """
        Lấy headers cho API request với token thích hợp.
        
        Args:
            token_index: Chỉ số token trong danh sách (nếu None, sẽ tự động chọn token tốt nhất)
            
        Returns:
            Headers cho API request
        """
        if not self.token_list:
            return {'Accept': 'application/vnd.github.v3+json'}
        
        if token_index is None:
            token_index = self._get_best_token_index()
        
        token = self.token_list[token_index % len(self.token_list)]
        self.current_token_index = token_index  # Cập nhật token hiện tại
        
        return {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {token}'
        }
    
    def _get_best_token_index(self) -> int:
        """
        Chọn token tốt nhất để sử dụng dựa trên rate limit còn lại.
        
        Returns:
            Chỉ số của token tốt nhất
        """
        # Nếu chỉ có 1 token hoặc không có token
        if len(self.token_list) <= 1:
            return 0
        
        current_time = time.time()
        best_token_index = 0
        best_remaining = -1
        
        for i, token in enumerate(self.token_list):
            # Kiểm tra xem token đã reset rate limit chưa
            if current_time > self.rate_limit_reset.get(token, 0):
                # Đã reset, đặt lại remaining cao
                self.rate_limit_remaining[token] = 5000
            
            # Chọn token có remaining cao nhất
            if self.rate_limit_remaining.get(token, 0) > best_remaining:
                best_remaining = self.rate_limit_remaining.get(token, 0)
                best_token_index = i
        
        return best_token_index
    
    def _update_rate_limit_info(self, token_index: int, response: requests.Response) -> None:
        """
        Cập nhật thông tin về rate limit từ response.
        
        Args:
            token_index: Chỉ số token đã sử dụng
            response: Response từ GitHub API
        """
        if token_index >= len(self.token_list):
            return
        
        token = self.token_list[token_index]
        self.api_calls[token] = self.api_calls.get(token, 0) + 1
        
        # Cập nhật thông tin rate limit từ headers
        remaining = response.headers.get('X-RateLimit-Remaining')
        reset = response.headers.get('X-RateLimit-Reset')
        
        if remaining is not None:
            self.rate_limit_remaining[token] = int(remaining)
        
        if reset is not None:
            self.rate_limit_reset[token] = int(reset)
    
    def get_commit_history(self, repo: str, branch: str = 'main', max_pages: int = 10, 
                          per_page: int = 100, token_index: Optional[int] = None) -> List[Dict]:
        """
        Lấy lịch sử commit từ repository.
        
        Args:
            repo: Tên repository (format: 'owner/repo')
            branch: Tên nhánh
            max_pages: Số trang tối đa cần lấy
            per_page: Số commit mỗi trang
            token_index: Chỉ số token để sử dụng
            
        Returns:
            List các commit
        """
        commits = []
        page = 1
        
        while page <= max_pages:
            token_index = token_index or self._get_best_token_index()
            headers = self._get_headers(token_index)
            
            url = f"{self.base_url}/repos/{repo}/commits"
            params = {
                'sha': branch,
                'per_page': per_page,
                'page': page
            }
            
            logger.info(f"Đang lấy commit từ {repo}, nhánh {branch}, trang {page} (token {token_index % len(self.token_list) + 1}/{len(self.token_list)})")
            response = requests.get(url, headers=headers, params=params)
            
            # Cập nhật thông tin rate limit
            self._update_rate_limit_info(token_index, response)
            
            if response.status_code == 200:
                page_commits = response.json()
                if not page_commits:
                    break  # Không còn commit
                
                commits.extend(page_commits)
                page += 1
                
                # Đợi một chút để tránh làm quá tải API
                time.sleep(0.1)
            elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                # Rate limit bị vượt quá, xử lý
                wait_time = self.handle_rate_limit(response, token_index)
                
                # Nếu có nhiều token, thử chuyển sang token khác
                if len(self.token_list) > 1:
                    next_token_index = (token_index + 1) % len(self.token_list)
                    logger.info(f"Chuyển sang token {next_token_index + 1}/{len(self.token_list)}")
                    token_index = next_token_index
            else:
                logger.error(f"Lỗi khi lấy commit: {response.status_code} - {response.text}")
                break
        
        return commits
    
    def get_commit_details(self, repo: str, commit_sha: str, token_index: Optional[int] = None) -> Dict:
        """
        Lấy chi tiết một commit cụ thể.
        
        Args:
            repo: Tên repository (format: 'owner/repo')
            commit_sha: SHA của commit
            token_index: Chỉ số token để sử dụng
            
        Returns:
            Chi tiết commit
        """
        token_index = token_index or self._get_best_token_index()
        headers = self._get_headers(token_index)
        
        url = f"{self.base_url}/repos/{repo}/commits/{commit_sha}"
        
        logger.debug(f"Đang lấy chi tiết commit {commit_sha} từ {repo}")
        response = requests.get(url, headers=headers)
        
        # Cập nhật thông tin rate limit
        self._update_rate_limit_info(token_index, response)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
            # Rate limit bị vượt quá, xử lý và thử lại
            self.handle_rate_limit(response, token_index)
            
            # Thử lại với token khác nếu có
            if len(self.token_list) > 1:
                next_token_index = (token_index + 1) % len(self.token_list)
                logger.info(f"Thử lại với token {next_token_index + 1}/{len(self.token_list)}")
                return self.get_commit_details(repo, commit_sha, next_token_index)
            
            return {}
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
    
    def handle_rate_limit(self, response: requests.Response, token_index: int) -> float:
        """
        Xử lý khi gặp rate limit từ GitHub API.
        
        Args:
            response: Response từ GitHub API
            token_index: Chỉ số token đã sử dụng
            
        Returns:
            Thời gian đã đợi (giây)
        """
        if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = max(reset_time - time.time(), 0) + 1
            
            # Cập nhật thông tin rate limit cho token
            if token_index < len(self.token_list):
                token = self.token_list[token_index]
                self.rate_limit_remaining[token] = 0
                self.rate_limit_reset[token] = reset_time
            
            # Chỉ đợi nếu tất cả các token đều đã hết rate limit
            all_tokens_exhausted = all(self.rate_limit_remaining.get(token, 0) < 10 for token in self.token_list)
            
            if all_tokens_exhausted:
                logger.warning(f"Tất cả token đều đã hết rate limit. Đợi {wait_time:.0f} giây.")
                time.sleep(wait_time)
                return wait_time
            else:
                logger.warning(f"Token {token_index+1} đã hết rate limit. Chuyển sang token khác.")
                return 0
        return 0
    
    def _process_repository(self, repo: str, max_commits: int, save_dir: str, token_index: int, chunk_id: int) -> Tuple[List[Dict], int]:
        """
        Xử lý một repository riêng lẻ và thu thập dữ liệu commit.
        
        Args:
            repo: Tên repository (format: 'owner/repo')
            max_commits: Số lượng commit tối đa cần lấy
            save_dir: Thư mục để lưu dữ liệu
            token_index: Chỉ số token để sử dụng
            chunk_id: ID của chunk cho file tạm
            
        Returns:
            Tuple chứa danh sách commit đã thu thập và số lượng commit đã xử lý
        """
        commits_collected = []
        commits_processed = 0
        
        try:
            # Xác định nhánh chính (main hoặc master)
            branch = self._detect_main_branch(repo, token_index)
            
            # Lấy danh sách commit
            commits = self.get_commit_history(
                repo=repo,
                branch=branch,
                max_pages=(max_commits // 100) + 1,
                per_page=100,
                token_index=token_index
            )
            
            # Giới hạn số lượng commit
            commits = commits[:max_commits]
            total_commits = len(commits)
            
            logger.info(f"Đã tìm thấy {total_commits} commit từ {repo}")
            
            # Xử lý từng commit
            for i, commit in enumerate(commits):
                commit_sha = commit.get('sha')
                if not commit_sha:
                    continue
                
                commits_processed += 1
                
                if i % 10 == 0:
                    logger.info(f"Đang xử lý commit {i+1}/{total_commits} từ {repo} (token {token_index + 1})")
                
                # Lấy chi tiết commit
                commit_detail = self.get_commit_details(repo, commit_sha, token_index)
                
                if commit_detail:
                    commit_data = self.extract_commit_data(commit_detail)
                    commit_data['metadata']['repository'] = repo
                    commits_collected.append(commit_data)
                
                # Lưu tạm sau mỗi 50 commit
                if len(commits_collected) % 50 == 0 and commits_collected:
                    self._save_intermediate_chunk(commits_collected, save_dir, [repo], chunk_id)
            
            return commits_collected, commits_processed
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý repository {repo}: {str(e)}")
            return commits_collected, commits_processed
    
    def _detect_main_branch(self, repo: str, token_index: int) -> str:
        """
        Phát hiện nhánh chính của repository (main hoặc master).
        
        Args:
            repo: Tên repository (format: 'owner/repo')
            token_index: Chỉ số token để sử dụng
            
        Returns:
            Tên nhánh chính
        """
        headers = self._get_headers(token_index)
        url = f"{self.base_url}/repos/{repo}"
        
        try:
            response = requests.get(url, headers=headers)
            self._update_rate_limit_info(token_index, response)
            
            if response.status_code == 200:
                repo_data = response.json()
                default_branch = repo_data.get('default_branch')
                if default_branch:
                    return default_branch
            
            # Thử các nhánh phổ biến
            branches = ['main', 'master', 'develop', 'trunk']
            for branch in branches:
                url = f"{self.base_url}/repos/{repo}/branches/{branch}"
                response = requests.get(url, headers=headers)
                self._update_rate_limit_info(token_index, response)
                
                if response.status_code == 200:
                    return branch
            
            # Mặc định nếu không tìm thấy
            return 'main'
            
        except Exception as e:
            logger.error(f"Lỗi khi phát hiện nhánh chính: {str(e)}")
            return 'main'
    
    def _save_intermediate_chunk(self, data: List[Dict], save_dir: str, repos: List[str], chunk_id: int) -> str:
        """
        Lưu một phần dữ liệu tạm thời.
        
        Args:
            data: Dữ liệu commit đã thu thập
            save_dir: Thư mục cơ sở để lưu dữ liệu
            repos: Danh sách repositories đã thu thập
            chunk_id: ID của chunk
            
        Returns:
            Đường dẫn đã lưu
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, f"temp_commits_{chunk_id}_{timestamp}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_samples': len(data),
                    'created_at': datetime.now().isoformat(),
                    'repositories': repos,
                    'is_complete': False,
                    'chunk_id': chunk_id
                },
                'data': data
            }, f, ensure_ascii=False)
        
        logger.debug(f"Đã lưu {len(data)} commit tạm thời vào {file_path}")
        return file_path
    
    def _save_final_data(self, data: List[Dict], output_path: str, repos: List[str]) -> None:
        """
        Lưu dữ liệu cuối cùng.
        
        Args:
            data: Dữ liệu commit đã thu thập
            output_path: Đường dẫn để lưu dữ liệu
            repos: Danh sách repositories đã thu thập
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_samples': len(data),
                    'created_at': datetime.now().isoformat(),
                    'repositories': repos,
                    'is_complete': True
                },
                'data': data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Đã lưu {len(data)} commit vào {output_path}")
    
    def collect_parallel(
        self, 
        repos: List[str], 
        output_path: str, 
        max_commits_per_repo: int = 1000,
        max_workers: int = 4
    ) -> None:
        """
        Thu thập dữ liệu commit từ nhiều repositories song song.
        
        Args:
            repos: Danh sách repositories (format: 'owner/repo')
            output_path: Đường dẫn để lưu dữ liệu
            max_commits_per_repo: Số lượng commit tối đa cần lấy từ mỗi repo
            max_workers: Số lượng worker tối đa để xử lý song song
        """
        all_commit_data = []
        total_processed = 0
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_commits")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Đảm bảo số lượng worker không vượt quá số lượng repos hoặc token
        max_workers = min(max_workers, len(repos), max(1, len(self.token_list)))
        
        logger.info(f"Bắt đầu thu thập song song với {max_workers} worker")
        
        # Tạo các nhiệm vụ xử lý
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Tạo các future cho mỗi repository
            future_to_repo = {}
            for i, repo in enumerate(repos):
                token_index = i % len(self.token_list) if self.token_list else 0
                future = executor.submit(
                    self._process_repository,
                    repo,
                    max_commits_per_repo,
                    temp_dir,
                    token_index,
                    i
                )
                future_to_repo[future] = repo
            
            # Xử lý kết quả khi hoàn thành
            for future in concurrent.futures.as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    commits, processed = future.result()
                    all_commit_data.extend(commits)
                    total_processed += processed
                    logger.info(f"Đã hoàn thành xử lý {repo}: {len(commits)} commit thu thập được")
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý {repo}: {str(e)}")
        
        # Lưu dữ liệu cuối cùng
        self._save_final_data(all_commit_data, output_path, repos)
        
        # Hiển thị thống kê
        logger.info(f"Thu thập hoàn tất: {len(all_commit_data)} commit thu thập được từ {total_processed} commit đã xử lý")
        logger.info(f"Dữ liệu đã được lưu vào {output_path}")
    
    def collect_from_repositories_batch(
        self,
        repo_lists: List[List[str]],
        output_dir: str,
        max_commits_per_repo: int = 1000,
        max_workers: int = 4
    ) -> None:
        """
        Thu thập dữ liệu từ nhiều batch repository.
        
        Args:
            repo_lists: Danh sách các batch repository
            output_dir: Thư mục để lưu dữ liệu
            max_commits_per_repo: Số lượng commit tối đa cần lấy từ mỗi repo
            max_workers: Số lượng worker tối đa để xử lý song song
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, repo_batch in enumerate(repo_lists):
            logger.info(f"Bắt đầu xử lý batch {i+1}/{len(repo_lists)} với {len(repo_batch)} repositories")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"github_commits_batch{i+1}_{timestamp}.json")
            
            self.collect_parallel(
                repos=repo_batch,
                output_path=output_path,
                max_commits_per_repo=max_commits_per_repo,
                max_workers=max_workers
            )
            
            logger.info(f"Đã hoàn thành batch {i+1}")
    
    def collect_with_deduplication(
        self,
        repos: List[str],
        output_path: str,
        max_commits_per_repo: int = 1000,
        max_workers: int = 4
    ) -> None:
        """
        Thu thập dữ liệu với khả năng loại bỏ trùng lặp.
        
        Args:
            repos: Danh sách repositories
            output_path: Đường dẫn để lưu dữ liệu
            max_commits_per_repo: Số lượng commit tối đa cần lấy từ mỗi repo
            max_workers: Số lượng worker tối đa để xử lý song song
        """
        all_commit_data = []
        seen_commit_ids = set()
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_commits")
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info(f"Bắt đầu thu thập với khả năng loại bỏ trùng lặp")
        
        # Thu thập dữ liệu
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Tạo các future cho mỗi repository
            future_to_repo = {}
            for i, repo in enumerate(repos):
                token_index = i % len(self.token_list) if self.token_list else 0
                future = executor.submit(
                    self._process_repository,
                    repo,
                    max_commits_per_repo,
                    temp_dir,
                    token_index,
                    i
                )
                future_to_repo[future] = repo
            
            # Xử lý kết quả khi hoàn thành
            for future in concurrent.futures.as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    commits, processed = future.result()
                    
                    # Loại bỏ commit trùng lặp
                    unique_commits = []
                    for commit in commits:
                        commit_id = commit.get('metadata', {}).get('commit_id', '')
                        if commit_id and commit_id not in seen_commit_ids:
                            seen_commit_ids.add(commit_id)
                            unique_commits.append(commit)
                    
                    all_commit_data.extend(unique_commits)
                    logger.info(f"Đã hoàn thành {repo}: {len(unique_commits)}/{len(commits)} commit duy nhất")
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý {repo}: {str(e)}")
        
        # Lưu dữ liệu cuối cùng
        self._save_final_data(all_commit_data, output_path, repos)
        
        # Hiển thị thống kê
        logger.info(f"Thu thập hoàn tất: {len(all_commit_data)} commit duy nhất thu thập được")
        logger.info(f"Dữ liệu đã được lưu vào {output_path}")

# Hàm tiện ích để chia danh sách repositories thành các batch
def split_repos_into_batches(repos: List[str], batch_size: int) -> List[List[str]]:
    """
    Chia danh sách repositories thành các batch nhỏ hơn.
    
    Args:
        repos: Danh sách repository
        batch_size: Kích thước mỗi batch
        
    Returns:
        Danh sách các batch repository
    """
    return [repos[i:i + batch_size] for i in range(0, len(repos), batch_size)]
