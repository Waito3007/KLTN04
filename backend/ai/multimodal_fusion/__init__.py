"""
Multi-Modal Fusion Network for Commit Analysis
Mô hình kết hợp thông tin văn bản và metadata để phân tích commit
"""

__version__ = "1.0.0"
__author__ = "AI Team"

from .models.multimodal_fusion import MultiModalFusionNetwork
from .data_preprocessing.text_processor import TextProcessor
from .data_preprocessing.metadata_processor import MetadataProcessor
from .training.multitask_trainer import MultiTaskTrainer

__all__ = [
    "MultiModalFusionNetwork",
    "TextProcessor", 
    "MetadataProcessor",
    "MultiTaskTrainer"
]
