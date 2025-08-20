from abc import ABC, abstractmethod
from typing import Dict

class IAreaAnalysisService(ABC):
    @abstractmethod
    def predict_area(self, commit_data: Dict) -> str:
        """
        Dự đoán phạm vi công việc (dev area) dựa trên dữ liệu commit.

        Args:
            commit_data (Dict): Dữ liệu commit bao gồm commit_message, diff_content,
                               files_count, lines_added, lines_removed, total_changes.

        Returns:
            str: Phạm vi công việc được dự đoán.
        """
        pass
