"""
Module dự đoán cho commit messages.
"""
import os
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

from models.multimodal_fusion_model import EnhancedMultimodalFusionModel
from data_processing.text_processor import TextProcessor
from data_processing.metadata_processor import MetadataProcessor
from data_processing.commit_processor import CommitDataProcessor

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommitPredictor:
    """Class dự đoán cho commit messages."""
    
    def __init__(
        self,
        model_path: str,
        text_processor_path: str,
        metadata_processor_path: str,
        device: Optional[str] = None
    ):
        """
        Khởi tạo predictor.
        
        Args:
            model_path: Đường dẫn đến file checkpoint của mô hình
            text_processor_path: Đường dẫn đến file text processor
            metadata_processor_path: Đường dẫn đến file metadata processor
            device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
        """
        # Xác định thiết bị
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Tải text processor
        self.text_processor = TextProcessor.load(text_processor_path)
        logger.info(f"Đã tải text processor từ {text_processor_path}")
        
        # Tải metadata processor
        self.metadata_processor = MetadataProcessor.load(metadata_processor_path)
        logger.info(f"Đã tải metadata processor từ {metadata_processor_path}")
        
        # Tải model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_config = checkpoint['model_config']
        
        self.model = EnhancedMultimodalFusionModel(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Đã tải model từ {model_path}")
        
        # Khởi tạo commit processor để trích xuất đặc trưng
        self.commit_processor = CommitDataProcessor("")
    
    def extract_features(self, commit_message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng từ commit message và metadata.
        
        Args:
            commit_message: Nội dung commit message
            metadata: Dict metadata (nếu None thì tạo metadata rỗng)
            
        Returns:
            Dict đặc trưng đã trích xuất
        """
        if metadata is None:
            metadata = {}
        
        # Trích xuất đặc trưng từ commit message
        conv_features = self.commit_processor.extract_conventional_commit_features(commit_message)
        nlp_features = self.commit_processor.extract_nlp_features(commit_message)
        
        # Trích xuất đặc trưng từ metadata
        meta_features = self.commit_processor.extract_metadata_features(metadata)
        
        # Kết hợp các đặc trưng
        features = {**conv_features, **nlp_features, **meta_features}
        
        return features
    
    def predict(self, commit_message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dự đoán cho một commit message.
        
        Args:
            commit_message: Nội dung commit message
            metadata: Dict metadata (nếu None thì tạo metadata rỗng)
            
        Returns:
            Dict kết quả dự đoán
        """
        # Trích xuất đặc trưng
        features = self.extract_features(commit_message, metadata)
        
        # Xử lý text
        text_tensor = torch.tensor(self.text_processor.process([commit_message])[0], dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Xử lý metadata
        metadata_tensor = torch.tensor(self.metadata_processor.process_single(features), dtype=torch.float).unsqueeze(0).to(self.device)
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(text_tensor, metadata_tensor)
        
        # Xử lý kết quả
        results = {}
        
        for task_name, task_output in outputs.items():
            task_config = self.model_config['task_heads'][task_name]
            
            if task_config.get('type', 'classification') == 'classification':
                # Đối với phân lớp
                probs = torch.softmax(task_output, dim=1)[0].cpu().numpy()
                pred_class = torch.argmax(task_output, dim=1).item()
                
                # Chuyển đổi dự đoán thành nhãn dễ đọc
                label = self.get_readable_label(task_name, pred_class)
                
                results[task_name] = {
                    'class': pred_class,
                    'label': label,
                    'confidence': float(probs[pred_class]),
                    'probabilities': {i: float(p) for i, p in enumerate(probs)}
                }
            else:
                # Đối với hồi quy
                value = float(task_output.squeeze().cpu().numpy())
                results[task_name] = {
                    'value': value
                }
        
        return results
    
    def get_readable_label(self, task_name: str, class_index: int) -> str:
        """
        Chuyển đổi chỉ số lớp thành nhãn dễ đọc.
        
        Args:
            task_name: Tên task
            class_index: Chỉ số lớp
            
        Returns:
            Nhãn dễ đọc
        """
        labels_mapping = {
            'risk_prediction': ['Low', 'Medium', 'High'],
            'complexity_prediction': ['Simple', 'Moderate', 'Complex'],
            'hotspot_prediction': ['Low', 'Medium', 'High'],
            'urgency_prediction': ['Low', 'Medium', 'High'],
            'completeness_prediction': ['Partial', 'Complete', 'Final']
        }
        
        if task_name in labels_mapping and class_index < len(labels_mapping[task_name]):
            return labels_mapping[task_name][class_index]
        
        return f"Class {class_index}"
    
    def batch_predict(self, commit_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dự đoán cho một batch commit.
        
        Args:
            commit_data: List các Dict chứa thông tin commit
            
        Returns:
            List các Dict kết quả dự đoán
        """
        results = []
        
        for commit in commit_data:
            commit_message = commit.get('text', '')
            metadata = commit.get('metadata', {})
            
            prediction = self.predict(commit_message, metadata)
            
            results.append({
                'commit_id': metadata.get('commit_id', ''),
                'commit_message': commit_message,
                'prediction': prediction
            })
        
        return results
    
    def generate_recommendations(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Tạo các đề xuất dựa trên kết quả dự đoán.
        
        Args:
            prediction: Dict kết quả dự đoán
            
        Returns:
            List các Dict đề xuất
        """
        recommendations = []
        
        # Xác định mức độ rủi ro
        if 'risk_prediction' in prediction:
            risk = prediction['risk_prediction']
            if risk['class'] == 2:  # High
                recommendations.append({
                    'type': 'risk',
                    'priority': 'high',
                    'message': f'Commit này có rủi ro cao (Độ tin cậy: {risk["confidence"]:.2f}). Cần review kỹ và kiểm tra test coverage.'
                })
            elif risk['class'] == 1:  # Medium
                recommendations.append({
                    'type': 'risk',
                    'priority': 'medium',
                    'message': f'Commit này có rủi ro trung bình (Độ tin cậy: {risk["confidence"]:.2f}). Nên review cẩn thận.'
                })
        
        # Xác định độ phức tạp
        if 'complexity_prediction' in prediction:
            complexity = prediction['complexity_prediction']
            if complexity['class'] == 2:  # Complex
                recommendations.append({
                    'type': 'complexity',
                    'priority': 'medium',
                    'message': f'Commit này có độ phức tạp cao (Độ tin cậy: {complexity["confidence"]:.2f}). Có thể cần thêm thời gian để hiểu và review.'
                })
        
        # Xác định mức độ hoàn thiện
        if 'completeness_prediction' in prediction:
            completeness = prediction['completeness_prediction']
            if completeness['class'] == 0:  # Partial
                recommendations.append({
                    'type': 'completeness',
                    'priority': 'high',
                    'message': f'Commit này có vẻ chưa hoàn thiện (Độ tin cậy: {completeness["confidence"]:.2f}). Cần theo dõi thêm các commits tiếp theo.'
                })
        
        # Xác định điểm nóng
        if 'hotspot_prediction' in prediction:
            hotspot = prediction['hotspot_prediction']
            if hotspot['class'] >= 1:  # Medium or High
                priority = 'medium' if hotspot['class'] == 1 else 'high'
                recommendations.append({
                    'type': 'hotspot',
                    'priority': priority,
                    'message': f'Commit này thay đổi vùng code "hotspot" (Độ tin cậy: {hotspot["confidence"]:.2f}). Cần chú ý đến tác động tiềm ẩn.'
                })
        
        # Xác định mức độ khẩn cấp
        if 'urgency_prediction' in prediction:
            urgency = prediction['urgency_prediction']
            if urgency['class'] == 2:  # High
                recommendations.append({
                    'type': 'urgency',
                    'priority': 'high',
                    'message': f'Commit này có độ khẩn cấp cao (Độ tin cậy: {urgency["confidence"]:.2f}). Nên ưu tiên xử lý sớm.'
                })
        
        return recommendations


def load_predictor(
    model_path: str,
    text_processor_path: str,
    metadata_processor_path: str,
    device: Optional[str] = None
) -> CommitPredictor:
    """
    Tải và khởi tạo predictor.
    
    Args:
        model_path: Đường dẫn đến file checkpoint của mô hình
        text_processor_path: Đường dẫn đến file text processor
        metadata_processor_path: Đường dẫn đến file metadata processor
        device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
        
    Returns:
        CommitPredictor
    """
    return CommitPredictor(
        model_path=model_path,
        text_processor_path=text_processor_path,
        metadata_processor_path=metadata_processor_path,
        device=device
    )


if __name__ == "__main__":
    # Ví dụ sử dụng
    predictor = load_predictor(
        model_path="checkpoints/best_model.pt",
        text_processor_path="processors/text_processor.json",
        metadata_processor_path="processors/metadata_processor.json",
        device="cuda"
    )
    
    # Dự đoán cho một commit
    commit_message = "fix: resolve authentication bug in login module"
    metadata = {
        "author": "developer1",
        "files_changed": 3,
        "additions": 25,
        "deletions": 10,
        "total_changes": 35
    }
    
    prediction = predictor.predict(commit_message, metadata)
    print("Prediction:")
    for task, result in prediction.items():
        print(f"  {task}: {result}")
    
    # Tạo đề xuất
    recommendations = predictor.generate_recommendations(prediction)
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  [{rec['priority']}] {rec['type']}: {rec['message']}")
