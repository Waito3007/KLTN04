"""
Module đánh giá mô hình.
"""
import os
import json
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, mean_squared_error
)

from models.multimodal_fusion_model import EnhancedMultimodalFusionModel

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommitModelEvaluator:
    """Class đánh giá mô hình commit."""
    
    def __init__(
        self,
        model: EnhancedMultimodalFusionModel,
        test_loader: DataLoader,
        device: Optional[str] = None
    ):
        """
        Khởi tạo evaluator.
        
        Args:
            model: Mô hình cần đánh giá
            test_loader: DataLoader cho tập test
            device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
        """
        self.model = model
        self.test_loader = test_loader
        
        # Xác định thiết bị
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Chuyển mô hình tới thiết bị
        self.model.to(self.device)
        
        # Lấy danh sách các task
        self.task_names = list(self.model.task_heads.keys())
        
        # Lưu kết quả đánh giá
        self.evaluation_results = {}
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Đánh giá mô hình trên tập test.
        
        Returns:
            Dict kết quả đánh giá
        """
        logger.info(f"Đánh giá mô hình trên {self.device}")
        
        self.model.eval()
        
        # Lưu trữ dự đoán và nhãn thực tế
        all_predictions = {task: [] for task in self.task_names}
        all_labels = {task: [] for task in self.task_names}
        all_raw_texts = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Chuyển dữ liệu tới thiết bị
                text = batch['text'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = {task: batch['labels'][task].to(self.device) for task in self.task_names}
                raw_texts = batch['raw_text']
                
                # Forward pass
                outputs = self.model(text, metadata)
                
                # Lưu dự đoán và nhãn
                for task_name in self.task_names:
                    task_config = self.model.config['task_heads'][task_name]
                    
                    if task_config.get('type', 'classification') == 'classification':
                        # Lấy lớp dự đoán
                        predictions = torch.argmax(outputs[task_name], dim=1).cpu().numpy()
                    else:
                        # Đối với hồi quy, lấy giá trị trực tiếp
                        predictions = outputs[task_name].squeeze().cpu().numpy()
                    
                    all_predictions[task_name].extend(predictions.tolist() if isinstance(predictions, np.ndarray) else [predictions])
                    all_labels[task_name].extend(labels[task_name].cpu().numpy().tolist())
                
                all_raw_texts.extend(raw_texts)
        
        # Tính toán các metrics và lưu kết quả
        results = {}
        
        for task_name in self.task_names:
            task_config = self.model.config['task_heads'][task_name]
            task_results = {}
            
            if task_config.get('type', 'classification') == 'classification':
                # Đối với phân lớp
                y_true = np.array(all_labels[task_name])
                y_pred = np.array(all_predictions[task_name])
                
                # Các metrics cơ bản
                task_results['accuracy'] = accuracy_score(y_true, y_pred)
                task_results['precision'] = precision_score(y_true, y_pred, average='weighted')
                task_results['recall'] = recall_score(y_true, y_pred, average='weighted')
                task_results['f1'] = f1_score(y_true, y_pred, average='weighted')
                
                # Confusion matrix và classification report
                task_results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
                task_results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
                
                # Log kết quả
                logger.info(f"Task: {task_name}")
                logger.info(f"  Accuracy: {task_results['accuracy']:.4f}")
                logger.info(f"  Precision: {task_results['precision']:.4f}")
                logger.info(f"  Recall: {task_results['recall']:.4f}")
                logger.info(f"  F1 Score: {task_results['f1']:.4f}")
            else:
                # Đối với hồi quy
                y_true = np.array(all_labels[task_name])
                y_pred = np.array(all_predictions[task_name])
                
                # Các metrics cơ bản
                task_results['mse'] = mean_squared_error(y_true, y_pred)
                task_results['rmse'] = np.sqrt(task_results['mse'])
                task_results['mae'] = np.mean(np.abs(y_true - y_pred))
                
                # Log kết quả
                logger.info(f"Task: {task_name}")
                logger.info(f"  MSE: {task_results['mse']:.4f}")
                logger.info(f"  RMSE: {task_results['rmse']:.4f}")
                logger.info(f"  MAE: {task_results['mae']:.4f}")
            
            results[task_name] = task_results
        
        # Lưu raw predictions để phân tích chi tiết
        results['raw'] = {
            'predictions': all_predictions,
            'labels': all_labels,
            'texts': all_raw_texts
        }
        
        self.evaluation_results = results
        
        return results
    
    def save_results(self, output_dir: str) -> None:
        """
        Lưu kết quả đánh giá.
        
        Args:
            output_dir: Thư mục lưu kết quả
        """
        if not self.evaluation_results:
            logger.warning("Không có kết quả đánh giá để lưu. Hãy gọi phương thức evaluate trước.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu kết quả tổng hợp dưới dạng JSON
        summary = {}
        for task_name, task_results in self.evaluation_results.items():
            if task_name == 'raw':
                continue
            
            task_config = self.model.config['task_heads'][task_name]
            if task_config.get('type', 'classification') == 'classification':
                summary[task_name] = {
                    'accuracy': task_results['accuracy'],
                    'precision': task_results['precision'],
                    'recall': task_results['recall'],
                    'f1': task_results['f1']
                }
            else:
                summary[task_name] = {
                    'mse': task_results['mse'],
                    'rmse': task_results['rmse'],
                    'mae': task_results['mae']
                }
        
        with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Lưu dự đoán chi tiết dưới dạng CSV
        raw_results = self.evaluation_results['raw']
        df_data = {'text': raw_results['texts']}
        
        for task_name in self.task_names:
            df_data[f'{task_name}_true'] = raw_results['labels'][task_name]
            df_data[f'{task_name}_pred'] = raw_results['predictions'][task_name]
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        # Vẽ confusion matrix cho các task phân lớp
        for task_name in self.task_names:
            task_config = self.model.config['task_heads'][task_name]
            if task_config.get('type', 'classification') == 'classification':
                cm = np.array(self.evaluation_results[task_name]['confusion_matrix'])
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {task_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(output_dir, f'confusion_matrix_{task_name}.png'))
                plt.close()
        
        logger.info(f"Đã lưu kết quả đánh giá vào {output_dir}")
    
    def analyze_errors(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Phân tích lỗi dự đoán.
        
        Args:
            output_dir: Thư mục lưu kết quả phân tích (nếu None thì không lưu)
            
        Returns:
            Dict kết quả phân tích
        """
        if not self.evaluation_results:
            logger.warning("Không có kết quả đánh giá để phân tích. Hãy gọi phương thức evaluate trước.")
            return {}
        
        raw_results = self.evaluation_results['raw']
        error_analysis = {}
        
        for task_name in self.task_names:
            task_config = self.model.config['task_heads'][task_name]
            
            if task_config.get('type', 'classification') == 'classification':
                y_true = np.array(raw_results['labels'][task_name])
                y_pred = np.array(raw_results['predictions'][task_name])
                texts = raw_results['texts']
                
                # Xác định các mẫu dự đoán sai
                incorrect_indices = np.where(y_true != y_pred)[0]
                incorrect_samples = []
                
                for idx in incorrect_indices:
                    incorrect_samples.append({
                        'text': texts[idx],
                        'true_label': int(y_true[idx]),
                        'predicted_label': int(y_pred[idx])
                    })
                
                # Thống kê lỗi theo loại
                error_counts = {}
                for sample in incorrect_samples:
                    error_type = f"{sample['true_label']} -> {sample['predicted_label']}"
                    if error_type not in error_counts:
                        error_counts[error_type] = 0
                    error_counts[error_type] += 1
                
                # Lưu kết quả phân tích
                error_analysis[task_name] = {
                    'total_errors': len(incorrect_indices),
                    'error_rate': len(incorrect_indices) / len(y_true),
                    'error_counts': error_counts,
                    'incorrect_samples': incorrect_samples
                }
                
                # Log kết quả
                logger.info(f"Phân tích lỗi cho task: {task_name}")
                logger.info(f"  Tổng số lỗi: {len(incorrect_indices)} / {len(y_true)} ({error_analysis[task_name]['error_rate']:.2%})")
                logger.info("  Phân bố lỗi:")
                for error_type, count in error_counts.items():
                    logger.info(f"    {error_type}: {count} ({count / len(incorrect_indices):.2%})")
        
        # Lưu kết quả nếu cần
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'error_analysis.json'), 'w', encoding='utf-8') as f:
                # Chỉ lưu một số mẫu lỗi đại diện để giảm kích thước file
                limited_analysis = {}
                for task_name, analysis in error_analysis.items():
                    limited_analysis[task_name] = {
                        'total_errors': analysis['total_errors'],
                        'error_rate': analysis['error_rate'],
                        'error_counts': analysis['error_counts'],
                        'incorrect_samples': analysis['incorrect_samples'][:100]  # Giới hạn số lượng mẫu
                    }
                
                json.dump(limited_analysis, f, indent=2)
            
            # Vẽ biểu đồ phân bố lỗi
            for task_name, analysis in error_analysis.items():
                error_types = list(analysis['error_counts'].keys())
                error_values = list(analysis['error_counts'].values())
                
                plt.figure(figsize=(12, 8))
                bars = plt.barh(error_types, error_values)
                plt.title(f'Error Distribution - {task_name}')
                plt.xlabel('Number of Errors')
                plt.ylabel('Error Type (True -> Predicted)')
                
                # Thêm số lượng vào biểu đồ
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width}', ha='left', va='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'error_distribution_{task_name}.png'))
                plt.close()
            
            logger.info(f"Đã lưu kết quả phân tích lỗi vào {output_dir}")
        
        return error_analysis


def evaluate_multimodal_fusion_model(
    model: EnhancedMultimodalFusionModel,
    test_loader: DataLoader,
    output_dir: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Đánh giá mô hình fusion đa phương thức.
    
    Args:
        model: Mô hình cần đánh giá
        test_loader: DataLoader cho tập test
        output_dir: Thư mục lưu kết quả (nếu None thì không lưu)
        device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
        
    Returns:
        Dict kết quả đánh giá
    """
    # Tạo evaluator
    evaluator = CommitModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # Đánh giá model
    results = evaluator.evaluate()
    
    # Lưu kết quả nếu cần
    if output_dir:
        evaluator.save_results(output_dir)
        evaluator.analyze_errors(output_dir)
    
    return results


if __name__ == "__main__":
    ()
    # Ví dụ sử dụng
    # Giả sử đã có model, test_loader
    
    # model = ...
    # test_loader = ...
    
    # results = evaluate_multimodal_fusion_model(
    #     model=model,
    #     test_loader=test_loader,
    #     output_dir='evaluation_results',
    #     device='cuda'
    # )
