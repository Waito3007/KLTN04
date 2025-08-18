from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class IMultiFusionCommitAnalystService(ABC):
    @abstractmethod
    def get_branches(self, repo_id: int):
        """
        Trả về danh sách branches thực tế từ DB.

        Args:
            repo_id (int): ID của repository.

        Returns:
            List[Dict[str, Any]]: Danh sách các branch.
        """
        pass

    @abstractmethod
    def get_all_repo_commits_raw(self, repo_id: int):
        """
        Trả về danh sách tất cả commits thực tế từ DB.

        Args:
            repo_id (int): ID của repository.

        Returns:
            List[Dict[str, Any]]: Danh sách các commit.
        """
        pass

    @abstractmethod
    async def get_all_repo_commits_with_analysis(self, repo_id: int, limit: int, offset: int, branch_name: Optional[str]) -> Dict[str, Any]:
        """
        Gets all repository commits with AI analysis.

        Args:
            repo_id (int): ID của repository.
            limit (int): Số lượng commit trên mỗi trang.
            offset (int): Vị trí bắt đầu của trang.
            branch_name (Optional[str]): Tên branch (nếu có).

        Returns:
            Dict[str, Any]: Kết quả phân tích commit.
        """
        pass
