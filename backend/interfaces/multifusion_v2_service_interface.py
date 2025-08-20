from abc import ABC, abstractmethod
from typing import List, Dict, Any

class IMultiFusionV2Service(ABC):
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về model.

        Returns:
            Dict[str, Any]: Thông tin model.
        """
        pass

    @abstractmethod
    def is_model_available(self) -> bool:
        """
        Kiểm tra xem model có sẵn hay không.

        Returns:
            bool: True nếu model sẵn sàng, False nếu không.
        """
        pass

    @abstractmethod
    def predict_commit_type_batch(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dự đoán loại commit cho một danh sách commit.

        Args:
            commits (List[Dict[str, Any]]): Danh sách commit.

        Returns:
            List[Dict[str, Any]]: Kết quả dự đoán.
        """
        pass
