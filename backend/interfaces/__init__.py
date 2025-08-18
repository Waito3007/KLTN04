"""
Interfaces module - Định nghĩa các interface cho design patterns
Tuân thủ nguyên tắc Dependency Inversion Principle của SOLID
"""

from .area_analysis_service_interface import IAreaAnalysisService
from .risk_analysis_service_interface import IRiskAnalysisService
from .task_service_interface import ITaskService

__all__ = [
    "IAreaAnalysisService",
    "IRiskAnalysisService", 
    "ITaskService"
]
