from abc import ABC, abstractmethod
from typing import Dict

class IRiskAnalysisService(ABC):
    """Interface cho Risk Analysis Service"""
    
    @abstractmethod
    def predict_risk(self, commit_data: Dict) -> str:
        """
        Dự đoán độ rủi ro của commit dựa trên dữ liệu commit.

        Args:
            commit_data (Dict): Dữ liệu commit bao gồm commit_message, diff_content,
                               files_count, lines_added, lines_removed, total_changes.

        Returns:
            str: Độ rủi ro được dự đoán ("lowrisk" hoặc "highrisk").
        """
        pass
