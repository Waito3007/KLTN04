"""
Module xử lý dữ liệu commit.
"""
import re
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommitDataProcessor:
    """Class xử lý dữ liệu commit để chuẩn bị cho huấn luyện."""
    
    def __init__(self, data_path: str):
        """
        Khởi tạo processor với đường dẫn đến file dữ liệu.
        
        Args:
            data_path: Đường dẫn đến file dữ liệu JSON
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Đọc dữ liệu từ file JSON."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Đã đọc {len(self.data.get('data', []))} mẫu từ {self.data_path}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
            self.data = {'metadata': {}, 'data': []}
    
    def extract_conventional_commit_features(self, commit_message: str) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng từ conventional commit format.
        
        Args:
            commit_message: Nội dung commit message
            
        Returns:
            Dict các đặc trưng conventional commit
        """
        # Conventional commit pattern: type(scope): description
        pattern = r'^(\w+)(?:\(([^)]+)\))?: (.+)'
        match = re.match(pattern, commit_message)
        
        features = {
            'is_conventional': 0,
            'commit_type': 'unknown',
            'has_scope': 0,
            'scope': '',
            'has_breaking_change': 0,
            'references_issue': 0,
            'is_revert': 0,
        }
        
        if match:
            commit_type, scope, description = match.groups()
            commit_type = commit_type.lower()
            
            features['is_conventional'] = 1
            features['commit_type'] = commit_type
            features['has_scope'] = 1 if scope else 0
            features['scope'] = scope if scope else ''
            
            # Trích xuất đặc trưng từ description
            features['has_breaking_change'] = 1 if 'BREAKING CHANGE' in commit_message else 0
            features['references_issue'] = 1 if re.search(r'#\d+', commit_message) else 0
            features['is_revert'] = 1 if commit_message.lower().startswith('revert') else 0
        else:
            # Không phải conventional commit format
            features['references_issue'] = 1 if re.search(r'#\d+', commit_message) else 0
            features['is_revert'] = 1 if commit_message.lower().startswith('revert') else 0
        
        return features
    
    def extract_nlp_features(self, commit_message: str) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng NLP cơ bản từ commit message.
        
        Args:
            commit_message: Nội dung commit message
            
        Returns:
            Dict các đặc trưng NLP
        """
        # Chuyển về chữ thường
        text = commit_message.lower()
        
        # Đếm số từ
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # Đếm số ký tự
        char_count = len(text)
        
        # Đếm số dòng
        line_count = len(text.split('\n'))
        
        # Đếm số từ đặc biệt
        bug_related = sum(1 for word in words if word in ['bug', 'fix', 'issue', 'error', 'crash', 'problem', 'fail'])
        feature_related = sum(1 for word in words if word in ['feature', 'add', 'new', 'implement', 'support', 'enhance'])
        test_related = sum(1 for word in words if word in ['test', 'unittest', 'spec', 'coverage'])
        doc_related = sum(1 for word in words if word in ['doc', 'docs', 'documentation', 'comment'])
        refactor_related = sum(1 for word in words if word in ['refactor', 'clean', 'reorganize', 'restructure'])
        
        # Đếm số dấu chấm câu
        punctuation_count = sum(1 for char in text if char in '.,;:!?')
        
        # Độ phức tạp văn bản (đơn giản)
        complexity = char_count / max(word_count, 1)
        
        # Phát hiện dấu hiệu WIP
        wip_indicators = ['wip', 'work in progress', 'in progress', 'not finished', 'incomplete']
        has_wip = 1 if any(indicator in text for indicator in wip_indicators) else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count,
            'bug_related_count': bug_related,
            'feature_related_count': feature_related,
            'test_related_count': test_related,
            'doc_related_count': doc_related,
            'refactor_related_count': refactor_related,
            'punctuation_count': punctuation_count,
            'text_complexity': complexity,
            'has_wip': has_wip
        }
    
    def extract_metadata_features(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trích xuất và chuyển đổi các đặc trưng từ metadata.
        
        Args:
            metadata: Dict metadata của commit
            
        Returns:
            Dict các đặc trưng từ metadata
        """
        features = {}
        
        # Lấy các trường cơ bản
        features['files_changed'] = metadata.get('files_changed', 0)
        features['additions'] = metadata.get('additions', 0)
        features['deletions'] = metadata.get('deletions', 0)
        features['total_changes'] = metadata.get('total_changes', 0)
        features['is_merge'] = 1 if metadata.get('is_merge', False) else 0
        
        # Tính tỷ lệ thêm/xóa
        if features['deletions'] > 0:
            features['add_del_ratio'] = features['additions'] / features['deletions']
        else:
            features['add_del_ratio'] = features['additions'] if features['additions'] > 0 else 0
        
        # Tính kích thước trung bình của thay đổi trên mỗi file
        if features['files_changed'] > 0:
            features['avg_change_per_file'] = features['total_changes'] / features['files_changed']
        else:
            features['avg_change_per_file'] = 0
        
        # Xử lý thời gian
        if 'timestamp' in metadata and metadata['timestamp']:
            try:
                dt = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                features['hour_of_day'] = dt.hour
                features['day_of_week'] = dt.weekday()  # 0 = Monday, 6 = Sunday
                features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            except:
                features['hour_of_day'] = 0
                features['day_of_week'] = 0
                features['is_weekend'] = 0
        else:
            features['hour_of_day'] = 0
            features['day_of_week'] = 0
            features['is_weekend'] = 0
        
        # Xử lý file types
        file_types = metadata.get('file_types', {})
        if file_types:
            features['has_py_files'] = 1 if '.py' in file_types else 0
            features['has_js_files'] = 1 if '.js' in file_types or '.jsx' in file_types or '.ts' in file_types or '.tsx' in file_types else 0
            features['has_html_files'] = 1 if '.html' in file_types or '.htm' in file_types else 0
            features['has_css_files'] = 1 if '.css' in file_types or '.scss' in file_types or '.sass' in file_types else 0
            features['has_config_files'] = 1 if any(ext in file_types for ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']) else 0
            features['has_doc_files'] = 1 if any(ext in file_types for ext in ['.md', '.rst', '.txt', '.pdf', '.doc', '.docx']) else 0
            features['has_test_files'] = 1 if any('test' in f.lower() for f in metadata.get('modified_files', [])) else 0
        else:
            features['has_py_files'] = 0
            features['has_js_files'] = 0
            features['has_html_files'] = 0
            features['has_css_files'] = 0
            features['has_config_files'] = 0
            features['has_doc_files'] = 0
            features['has_test_files'] = 0
        
        return features
    
    def auto_label_risk(self, commit_data: Dict[str, Any]) -> int:
        """
        Tự động gán nhãn rủi ro cho commit dựa trên các quy tắc.
        
        Args:
            commit_data: Dict dữ liệu commit đã xử lý
            
        Returns:
            Mức độ rủi ro (0=low, 1=medium, 2=high)
        """
        text = commit_data.get('text', '').lower()
        features = commit_data.get('features', {})
        
        # Các từ khóa liên quan đến rủi ro cao
        high_risk_keywords = ['critical', 'urgent', 'security', 'vulnerability', 'crash', 'emergency', 'fix critical']
        
        # Các từ khóa liên quan đến rủi ro trung bình
        medium_risk_keywords = ['bug', 'fix', 'issue', 'error', 'problem', 'resolve', 'workaround']
        
        # Các dấu hiệu từ metadata
        is_large_change = features.get('files_changed', 0) > 10 or features.get('total_changes', 0) > 500
        affects_many_areas = len(features.get('modified_directories', {})) > 5
        is_weekend_commit = features.get('is_weekend', 0) == 1
        is_late_night = features.get('hour_of_day', 0) < 6 or features.get('hour_of_day', 0) > 22
        
        # Kiểm tra các quy tắc
        if any(keyword in text for keyword in high_risk_keywords) or is_large_change:
            return 2  # High
        elif any(keyword in text for keyword in medium_risk_keywords) or affects_many_areas or is_weekend_commit or is_late_night:
            return 1  # Medium
        else:
            return 0  # Low
    
    def auto_label_complexity(self, commit_data: Dict[str, Any]) -> int:
        """
        Tự động gán nhãn độ phức tạp cho commit dựa trên các quy tắc.
        
        Args:
            commit_data: Dict dữ liệu commit đã xử lý
            
        Returns:
            Mức độ phức tạp (0=simple, 1=moderate, 2=complex)
        """
        text = commit_data.get('text', '').lower()
        features = commit_data.get('features', {})
        
        # Các từ khóa liên quan đến độ phức tạp cao
        complex_keywords = ['refactor', 'architecture', 'redesign', 'rewrite', 'complex', 'complicated']
        
        # Các từ khóa liên quan đến độ phức tạp trung bình
        moderate_keywords = ['algorithm', 'performance', 'optimize', 'enhance', 'improve']
        
        # Các dấu hiệu từ metadata
        is_large_change = features.get('files_changed', 0) > 5 or features.get('total_changes', 0) > 200
        affects_multiple_areas = len(features.get('modified_directories', {})) > 3
        
        # Kiểm tra các quy tắc
        if any(keyword in text for keyword in complex_keywords) or (is_large_change and affects_multiple_areas):
            return 2  # Complex
        elif any(keyword in text for keyword in moderate_keywords) or is_large_change or affects_multiple_areas:
            return 1  # Moderate
        else:
            return 0  # Simple
    
    def auto_label_hotspot(self, commit_data: Dict[str, Any]) -> int:
        """
        Tự động gán nhãn điểm nóng cho commit dựa trên các quy tắc.
        
        Args:
            commit_data: Dict dữ liệu commit đã xử lý
            
        Returns:
            Mức độ điểm nóng (0=low, 1=medium, 2=high)
        """
        text = commit_data.get('text', '').lower()
        features = commit_data.get('features', {})
        
        # Các từ khóa liên quan đến điểm nóng
        hotspot_keywords = ['hotfix', 'quickfix', 'bandaid', 'patch', 'urgent fix', 'emergency']
        
        # Các dấu hiệu từ metadata
        is_frequent_change = features.get('files_changed', 0) > 0  # Cần thông tin lịch sử thực tế
        
        # Giả lập một số quy tắc cho ví dụ
        is_core_file = any('core' in f.lower() for f in commit_data.get('metadata', {}).get('modified_files', []))
        is_config_file = features.get('has_config_files', 0) == 1
        
        # Kiểm tra các quy tắc
        if any(keyword in text for keyword in hotspot_keywords) or is_core_file:
            return 2  # High
        elif is_frequent_change or is_config_file:
            return 1  # Medium
        else:
            return 0  # Low
    
    def auto_label_urgency(self, commit_data: Dict[str, Any]) -> int:
        """
        Tự động gán nhãn độ khẩn cấp cho commit dựa trên các quy tắc.
        
        Args:
            commit_data: Dict dữ liệu commit đã xử lý
            
        Returns:
            Mức độ khẩn cấp (0=low, 1=medium, 2=high)
        """
        text = commit_data.get('text', '').lower()
        features = commit_data.get('features', {})
        
        # Các từ khóa liên quan đến độ khẩn cấp cao
        urgent_keywords = ['urgent', 'asap', 'emergency', 'critical', 'immediate', 'hotfix', 'production issue']
        
        # Các từ khóa liên quan đến độ khẩn cấp trung bình
        medium_keywords = ['important', 'fix', 'issue', 'bug', 'needed', 'soon']
        
        # Các dấu hiệu từ metadata
        is_security_related = 'security' in text or any('security' in f.lower() for f in commit_data.get('metadata', {}).get('modified_files', []))
        is_auth_related = 'auth' in text or any('auth' in f.lower() for f in commit_data.get('metadata', {}).get('modified_files', []))
        
        # Kiểm tra các quy tắc
        if any(keyword in text for keyword in urgent_keywords) or is_security_related:
            return 2  # High
        elif any(keyword in text for keyword in medium_keywords) or is_auth_related:
            return 1  # Medium
        else:
            return 0  # Low
    
    def auto_label_completeness(self, commit_data: Dict[str, Any]) -> int:
        """
        Tự động gán nhãn mức độ hoàn thiện cho commit dựa trên các quy tắc.
        
        Args:
            commit_data: Dict dữ liệu commit đã xử lý
            
        Returns:
            Mức độ hoàn thiện (0=partial, 1=complete, 2=final)
        """
        text = commit_data.get('text', '').lower()
        features = commit_data.get('features', {})
        
        # Các từ khóa liên quan đến "work in progress"
        wip_indicators = ['wip', 'work in progress', 'ongoing', 'partial', 'incomplete', 
                         'not finished', 'in progress', 'draft', 'to be continued']
        
        # Các từ khóa liên quan đến hoàn thành
        completion_indicators = ['complete', 'completed', 'finish', 'finished', 'done', 
                               'resolved', 'fixes', 'closes', 'implements', 'resolves']
        
        # Các từ khóa liên quan đến hoàn thiện cuối cùng
        final_indicators = ['final', 'finalize', 'finalized', 'ready for review', 
                          'ready for merge', 'ready for production']
        
        # Kiểm tra các quy tắc
        if any(indicator in text for indicator in wip_indicators) or features.get('has_wip', 0) == 1:
            return 0  # Partial
        elif any(indicator in text for indicator in final_indicators):
            return 2  # Final
        elif any(indicator in text for indicator in completion_indicators):
            return 1  # Complete
        else:
            return 1  # Mặc định là Complete
    
    def process_data(self) -> List[Dict[str, Any]]:
        """
        Xử lý toàn bộ dữ liệu và gán nhãn tự động.
        
        Returns:
            List các mẫu dữ liệu đã xử lý
        """
        processed_samples = []
        
        for sample in self.data.get('data', []):
            try:
                text = sample.get('text', '')
                metadata = sample.get('metadata', {})
                
                # Trích xuất đặc trưng
                conv_features = self.extract_conventional_commit_features(text)
                nlp_features = self.extract_nlp_features(text)
                meta_features = self.extract_metadata_features(metadata)
                
                # Kết hợp các đặc trưng
                features = {**conv_features, **nlp_features, **meta_features}
                
                # Tạo mẫu đã xử lý
                processed_sample = {
                    'text': text,
                    'features': features,
                    'metadata': metadata
                }
                
                # Gán nhãn tự động (trừ khi đã có nhãn)
                if 'labels' not in sample or not sample['labels']:
                    risk_level = self.auto_label_risk(processed_sample)
                    complexity_level = self.auto_label_complexity(processed_sample)
                    hotspot_level = self.auto_label_hotspot(processed_sample)
                    urgency_level = self.auto_label_urgency(processed_sample)
                    completeness_level = self.auto_label_completeness(processed_sample)
                    
                    processed_sample['labels'] = {
                        'risk_prediction': risk_level,
                        'complexity_prediction': complexity_level,
                        'hotspot_prediction': hotspot_level,
                        'urgency_prediction': urgency_level,
                        'completeness_prediction': completeness_level
                    }
                else:
                    processed_sample['labels'] = sample['labels']
                
                processed_samples.append(processed_sample)
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý mẫu: {str(e)}")
        
        self.processed_data = processed_samples
        logger.info(f"Đã xử lý {len(processed_samples)} mẫu")
        
        return processed_samples
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Lưu dữ liệu đã xử lý.
        
        Args:
            output_path: Đường dẫn để lưu dữ liệu
        """
        if not self.processed_data:
            logger.warning("Không có dữ liệu đã xử lý để lưu")
            return
        
        output_data = {
            'metadata': {
                'total_samples': len(self.processed_data),
                'created_at': datetime.now().isoformat(),
                'source_dataset': self.data.get('metadata', {}).get('repositories', []),
                'features': list(self.processed_data[0]['features'].keys()) if self.processed_data else []
            },
            'data': self.processed_data
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Đã lưu {len(self.processed_data)} mẫu đã xử lý vào {output_path}")
    
    def create_train_val_test_split(self, processed_data_path: str, output_dir: str, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42) -> None:
        """
        Chia dữ liệu thành tập train, validation và test.
        
        Args:
            processed_data_path: Đường dẫn đến file dữ liệu đã xử lý
            output_dir: Thư mục để lưu các tập dữ liệu
            test_size: Tỷ lệ dữ liệu test
            val_size: Tỷ lệ dữ liệu validation
            random_state: Seed cho quá trình chia ngẫu nhiên
        """
        # Đọc dữ liệu đã xử lý
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        data_samples = processed_data.get('data', [])
        
        # Chia dữ liệu
        train_samples, test_samples = train_test_split(data_samples, test_size=test_size, random_state=random_state)
        train_samples, val_samples = train_test_split(train_samples, test_size=val_size/(1-test_size), random_state=random_state)
        
        # Tạo metadata cho các tập dữ liệu
        train_data = {
            'metadata': {**processed_data.get('metadata', {}), 'split': 'train', 'samples': len(train_samples)},
            'data': train_samples
        }
        
        val_data = {
            'metadata': {**processed_data.get('metadata', {}), 'split': 'validation', 'samples': len(val_samples)},
            'data': val_samples
        }
        
        test_data = {
            'metadata': {**processed_data.get('metadata', {}), 'split': 'test', 'samples': len(test_samples)},
            'data': test_samples
        }
        
        # Lưu các tập dữ liệu
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, 'val.json'), 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Đã chia và lưu dữ liệu: {len(train_samples)} train, {len(val_samples)} validation, {len(test_samples)} test")


if __name__ == "__main__":
    # Ví dụ sử dụng
    processor = CommitDataProcessor("raw_github_commits.json")
    processor.process_data()
    processor.save_processed_data("processed_commits.json")
    processor.create_train_val_test_split(
        processed_data_path="processed_commits.json",
        output_dir="datasets",
        test_size=0.15,
        val_size=0.15
    )
