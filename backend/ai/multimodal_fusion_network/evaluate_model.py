import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from data_processing.metadata_processor import MetadataProcessor
from data_processing.text_processor import TextProcessor
from data_processing.label_processor import LabelProcessor
from model.multimodal_commit_classifier import MultimodalCommitClassifier
from utils.data_loader import CommitDataset
from torch.utils.data import DataLoader
import argparse

def evaluate_model(model_path, test_path, batch_size=32, output_dir=None):
    """Đánh giá mô hình trên tập test và lưu báo cáo kết quả."""
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # Tải các processor
    processor_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'processors')
    metadata_processor = MetadataProcessor.load(os.path.join(processor_dir, 'metadata_processor.json'))
    text_processor = TextProcessor.load(os.path.join(processor_dir, 'text_processor.json'))
    label_processor = LabelProcessor.load(os.path.join(processor_dir, 'label_processor.json'))
    
    # Tạo dataset test
    test_dataset = CommitDataset(test_path, text_processor, metadata_processor, label_processor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Tải model từ checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Tạo model với cấu trúc giống khi train
    model_config = checkpoint['model_config']
    model = MultimodalCommitClassifier(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Đang đánh giá mô hình...")
    
    # Thực hiện dự đoán
    all_predictions = {}
    all_labels = {}
    label_keys = label_processor.get_label_keys()
    
    with torch.no_grad():
        for batch in test_loader:
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            outputs = model(text_ids, text_mask, metadata)
            
            # Thu thập predictions và labels thực tế
            for key in label_keys:
                if key not in all_predictions:
                    all_predictions[key] = []
                    all_labels[key] = []
                
                pred = outputs[key]
                if label_processor.is_multilabel(key):
                    pred_labels = (torch.sigmoid(pred) > 0.5).float()
                else:
                    pred_labels = torch.argmax(pred, dim=1)
                
                all_predictions[key].append(pred_labels.cpu().numpy())
                all_labels[key].append(labels[key].cpu().numpy())
    
    # Kết hợp tất cả batch
    for key in label_keys:
        all_predictions[key] = np.vstack(all_predictions[key])
        all_labels[key] = np.vstack(all_labels[key])
    
    # Tính toán và lưu metrics
    results = {}
    confusion_matrices = {}
    
    for key in label_keys:
        print(f"\nĐánh giá trường nhãn: {key}")
        
        if label_processor.is_multilabel(key):
            # Metrics cho multi-label
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[key], all_predictions[key], average='micro'
            )
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                all_labels[key], all_predictions[key], average='macro'
            )
            
            # Tính accuracy (exact match)
            accuracy = np.mean(np.all(all_predictions[key] == all_labels[key], axis=1))
            
            print(f"Accuracy (exact match): {accuracy:.4f}")
            print(f"Precision (micro): {precision:.4f}")
            print(f"Recall (micro): {recall:.4f}")
            print(f"F1 Score (micro): {f1:.4f}")
            print(f"Precision (macro): {precision_macro:.4f}")
            print(f"Recall (macro): {recall_macro:.4f}")
            print(f"F1 Score (macro): {f1_macro:.4f}")
            
            results[key] = {
                'accuracy': float(accuracy),
                'precision_micro': float(precision),
                'recall_micro': float(recall),
                'f1_micro': float(f1),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro)
            }
            
            # Vẽ heatmap cho từng lớp
            if all_predictions[key].shape[1] <= 10:  # Chỉ vẽ nếu số lớp không quá nhiều
                class_names = label_processor.get_class_names(key)
                
                # Tính tỉ lệ dự đoán đúng cho từng class
                class_accuracy = {}
                for i in range(all_predictions[key].shape[1]):
                    class_acc = np.mean(all_predictions[key][:, i] == all_labels[key][:, i])
                    class_accuracy[class_names[i]] = class_acc
                    
                plt.figure(figsize=(10, 6))
                sns.barplot(x=list(class_accuracy.keys()), y=list(class_accuracy.values()))
                plt.title(f'Accuracy by Class - {key}')
                plt.xlabel('Classes')
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'accuracy_by_class_{key}.png'))
                plt.close()
                
                # Lưu class accuracy vào results
                results[key]['class_accuracy'] = class_accuracy
        else:
            # Metrics cho single-label
            y_true = np.argmax(all_labels[key], axis=1)
            y_pred = np.argmax(all_predictions[key], axis=1)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro'
            )
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            results[key] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
            # Tạo confusion matrix
            class_names = label_processor.get_class_names(key)
            cm = confusion_matrix(y_true, y_pred)
            
            # Vẽ confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {key}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{key}.png'))
            plt.close()
            
            confusion_matrices[key] = cm.tolist()
    
    # Tạo báo cáo chi tiết
    report = {
        'model_path': model_path,
        'test_path': test_path,
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': results,
        'confusion_matrices': confusion_matrices
    }
    
    # Lưu báo cáo
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nĐánh giá hoàn tất! Báo cáo đã được lưu tại: {report_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Đánh giá mô hình multimodal phân tích commit')
    parser.add_argument('--model_path', type=str, 
                        default='E:/Dự Án Của Nghĩa/KLTN04/backend/ai/multimodal_fusion_network/commit_analysis/checkpoints/training_20250628_155110/best_model.pt',
                        help='Đường dẫn tới checkpoint của model')
    parser.add_argument('--test_path', type=str, 
                        default='data/processed/test.json',
                        help='Đường dẫn tới tập dữ liệu test')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size cho đánh giá')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Thư mục lưu kết quả đánh giá (mặc định: thư mục chứa model/evaluation)')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_path, args.batch_size, args.output_dir)