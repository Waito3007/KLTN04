"""
Metadata Processor for Multi-Modal Fusion Network
Xử lý và chuẩn bị metadata từ GitHub commits
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timezone
import re
from pathlib import Path

class MetadataProcessor:
    """
    Lớp xử lý metadata cho GitHub commits
    Bao gồm commit stats, file info, author info, timestamp info
    """
    
    def __init__(self, normalize_features: bool = True, 
                 categorical_method: str = "embedding",  # "embedding", "onehot"
                 max_files: int = 50,
                 max_authors: int = 1000):
        """
        Args:
            normalize_features: Có chuẩn hóa features số không
            categorical_method: Phương pháp encode categorical ("embedding", "onehot")
            max_files: Số file tối đa để track
            max_authors: Số author tối đa để track
        """
        self.normalize_features = normalize_features
        self.categorical_method = categorical_method
        self.max_files = max_files
        self.max_authors = max_authors
        
        # Scalers for numerical features
        self.numerical_scaler = StandardScaler()
        self.ratio_scaler = MinMaxScaler()
        
        # Encoders for categorical features
        self.file_type_encoder = LabelEncoder()
        self.author_encoder = LabelEncoder()
        self.file_path_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        
        # Feature statistics
        self.feature_stats = {}
        self.is_fitted = False
        
    def extract_file_features(self, files_data: List[Dict]) -> Dict[str, Any]:
        """
        Trích xuất features từ thông tin files
        
        Args:
            files_data: List of file info dicts với keys: filename, status, additions, deletions, changes
        """
        if not files_data:
            return self._get_empty_file_features()
        
        features = {}
        
        # Basic file stats
        features['num_files'] = len(files_data)
        features['total_additions'] = sum(f.get('additions', 0) for f in files_data)
        features['total_deletions'] = sum(f.get('deletions', 0) for f in files_data)
        features['total_changes'] = sum(f.get('changes', 0) for f in files_data)
        
        # File types
        file_extensions = []
        file_paths = []
        for file_info in files_data:
            filename = file_info.get('filename', '')
            if filename:
                file_paths.append(filename)
                # Extract extension
                if '.' in filename:
                    ext = filename.split('.')[-1].lower()
                    file_extensions.append(ext)
        
        # File type diversity
        unique_extensions = list(set(file_extensions))
        features['num_file_types'] = len(unique_extensions)
        features['file_types'] = unique_extensions[:10]  # Top 10 types
        
        # File depth analysis
        depths = []
        for path in file_paths:
            depth = len(Path(path).parts) - 1  # Subtract 1 for filename
            depths.append(depth)
        
        if depths:
            features['avg_file_depth'] = np.mean(depths)
            features['max_file_depth'] = np.max(depths)
            features['min_file_depth'] = np.min(depths)
        else:
            features['avg_file_depth'] = 0
            features['max_file_depth'] = 0
            features['min_file_depth'] = 0
        
        # Change distribution
        if features['total_changes'] > 0:
            features['additions_ratio'] = features['total_additions'] / features['total_changes']
            features['deletions_ratio'] = features['total_deletions'] / features['total_changes']
        else:
            features['additions_ratio'] = 0
            features['deletions_ratio'] = 0
        
        # File status analysis
        status_counts = {}
        for file_info in files_data:
            status = file_info.get('status', 'modified')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        features['added_files'] = status_counts.get('added', 0)
        features['modified_files'] = status_counts.get('modified', 0)
        features['deleted_files'] = status_counts.get('removed', 0)
        features['renamed_files'] = status_counts.get('renamed', 0)
        
        # Large file changes indicator
        large_changes = sum(1 for f in files_data if f.get('changes', 0) > 100)
        features['large_change_files'] = large_changes
        features['has_large_changes'] = large_changes > 0
        
        return features
    
    def _get_empty_file_features(self) -> Dict[str, Any]:
        """Trả về features mặc định khi không có file data"""
        return {
            'num_files': 0,
            'total_additions': 0,
            'total_deletions': 0,
            'total_changes': 0,
            'num_file_types': 0,
            'file_types': [],
            'avg_file_depth': 0,
            'max_file_depth': 0,
            'min_file_depth': 0,
            'additions_ratio': 0,
            'deletions_ratio': 0,            'added_files': 0,
            'modified_files': 0,
            'deleted_files': 0,
            'renamed_files': 0,
            'large_change_files': 0,
            'has_large_changes': False
        }
    
    def extract_author_features(self, author_info, commit_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Trích xuất features từ thông tin author
        
        Args:
            author_info: Dict với keys: login, name, email, etc. HOẶC string author name
            commit_history: Lịch sử commit gần đây của author (optional)
        """
        features = {}
        
        # Handle both dict and string input
        if isinstance(author_info, str):
            # Simple string author name
            features['author_login'] = author_info
            features['author_name'] = author_info
            features['author_email'] = ''
        else:
            # Dict author info
            features['author_login'] = author_info.get('login', 'unknown')
            features['author_name'] = author_info.get('name', '')
            features['author_email'] = author_info.get('email', '')
        
        # Author activity pattern (nếu có lịch sử)
        if commit_history:
            features['recent_commits_count'] = len(commit_history)
            
            # Tính average commit size
            recent_changes = [c.get('stats', {}).get('total', 0) for c in commit_history]
            features['avg_recent_commit_size'] = np.mean(recent_changes) if recent_changes else 0
            
            # Frequency pattern
            if len(commit_history) >= 2:
                timestamps = [c.get('timestamp') for c in commit_history if c.get('timestamp')]
                if len(timestamps) >= 2:
                    # Calculate time between commits
                    time_diffs = []
                    for i in range(1, len(timestamps)):
                        try:
                            t1 = datetime.fromisoformat(timestamps[i-1].replace('Z', '+00:00'))
                            t2 = datetime.fromisoformat(timestamps[i].replace('Z', '+00:00'))
                            diff_hours = abs((t2 - t1).total_seconds() / 3600)
                            time_diffs.append(diff_hours)
                        except:
                            continue
                    
                    if time_diffs:
                        features['avg_commit_interval_hours'] = np.mean(time_diffs)
                        features['commit_frequency_score'] = min(24 / np.mean(time_diffs), 10) if np.mean(time_diffs) > 0 else 0
                    else:
                        features['avg_commit_interval_hours'] = 24
                        features['commit_frequency_score'] = 1
                else:
                    features['avg_commit_interval_hours'] = 24
                    features['commit_frequency_score'] = 1
            else:
                features['avg_commit_interval_hours'] = 24
                features['commit_frequency_score'] = 1
        else:
            features['recent_commits_count'] = 0
            features['avg_recent_commit_size'] = 0
            features['avg_commit_interval_hours'] = 24
            features['commit_frequency_score'] = 1
        
        return features
    
    def extract_timestamp_features(self, timestamp: str) -> Dict[str, Any]:
        """
        Trích xuất features từ timestamp
        """
        features = {}
        
        try:
            # Parse timestamp
            if timestamp.endswith('Z'):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(timestamp)
            
            # Time-based features
            features['hour_of_day'] = dt.hour
            features['day_of_week'] = dt.weekday()  # 0 = Monday
            features['day_of_month'] = dt.day
            features['month'] = dt.month
            features['year'] = dt.year
            
            # Derived features
            features['is_weekend'] = dt.weekday() >= 5
            features['is_business_hours'] = 9 <= dt.hour <= 17
            features['is_late_night'] = dt.hour >= 22 or dt.hour <= 6
            
            # Season (approximate)
            if dt.month in [12, 1, 2]:
                features['season'] = 'winter'
            elif dt.month in [3, 4, 5]:
                features['season'] = 'spring'
            elif dt.month in [6, 7, 8]:
                features['season'] = 'summer'
            else:
                features['season'] = 'fall'
                
        except Exception as e:
            # Default values if parsing fails
            features.update({
                'hour_of_day': 12,
                'day_of_week': 0,
                'day_of_month': 1,
                'month': 1,
                'year': 2024,
                'is_weekend': False,
                'is_business_hours': True,
                'is_late_night': False,
                'season': 'spring'
            })
        
        return features
    
    def create_feature_engineering(self, file_features: Dict, author_features: Dict, timestamp_features: Dict) -> Dict[str, Any]:
        """
        Tạo các feature engineering phức tạp hơn
        """
        engineered = {}
        
        # Commit complexity score
        complexity_score = 0
        complexity_score += min(file_features['num_files'] / 10, 1.0) * 0.3  # File count impact
        complexity_score += min(file_features['total_changes'] / 1000, 1.0) * 0.4  # Change size impact
        complexity_score += min(file_features['num_file_types'] / 5, 1.0) * 0.2  # Diversity impact
        complexity_score += min(file_features['max_file_depth'] / 10, 1.0) * 0.1  # Depth impact
        engineered['complexity_score'] = complexity_score
        
        # Risk assessment
        risk_score = 0
        risk_score += file_features['has_large_changes'] * 0.3
        risk_score += (file_features['deleted_files'] / max(file_features['num_files'], 1)) * 0.2
        risk_score += min(author_features['commit_frequency_score'] / 5, 1.0) * 0.2  # High frequency = higher risk
        risk_score += timestamp_features['is_late_night'] * 0.1
        risk_score += (timestamp_features['is_weekend'] and not timestamp_features['is_business_hours']) * 0.2
        engineered['risk_score'] = min(risk_score, 1.0)
        
        # Urgency indicators
        urgency_score = 0
        urgency_score += timestamp_features['is_late_night'] * 0.4
        urgency_score += timestamp_features['is_weekend'] * 0.3
        urgency_score += (author_features['commit_frequency_score'] > 5) * 0.3
        engineered['urgency_score'] = min(urgency_score, 1.0)
        
        # Code churn metrics
        if file_features['total_changes'] > 0:
            churn_ratio = (file_features['total_additions'] + file_features['total_deletions']) / file_features['total_changes']
            engineered['code_churn'] = min(churn_ratio, 2.0)
        else:
            engineered['code_churn'] = 0
        
        # File type hotspots (common file types that often have issues)
        risky_extensions = ['js', 'ts', 'py', 'java', 'cpp', 'c', 'php']
        config_extensions = ['json', 'xml', 'yml', 'yaml', 'cfg', 'conf']
        doc_extensions = ['md', 'txt', 'rst', 'doc']
        
        engineered['touches_risky_files'] = any(ext in risky_extensions for ext in file_features['file_types'])
        engineered['touches_config_files'] = any(ext in config_extensions for ext in file_features['file_types'])
        engineered['touches_doc_files'] = any(ext in doc_extensions for ext in file_features['file_types'])
        
        return engineered
    
    def fit(self, metadata_samples: List[Dict]) -> None:
        """
        Fit các encoders và scalers với training data
        """
        print("🔧 Fitting metadata processors...")
        
        # Collect all features
        all_numerical_features = []
        all_categorical_features = {
            'authors': [],
            'file_types': [],
            'seasons': []
        }
        
        for sample in metadata_samples:
            # Extract features
            file_features = self.extract_file_features(sample.get('files', []))
            author_features = self.extract_author_features(
                sample.get('author', {}), 
                sample.get('commit_history', [])
            )
            timestamp_features = self.extract_timestamp_features(sample.get('timestamp', ''))
            engineered_features = self.create_feature_engineering(file_features, author_features, timestamp_features)
            
            # Collect numerical features
            numerical = self._get_numerical_features(file_features, author_features, timestamp_features, engineered_features)
            all_numerical_features.append(numerical)
            
            # Collect categorical features
            all_categorical_features['authors'].append(author_features['author_login'])
            all_categorical_features['file_types'].extend(file_features['file_types'])
            all_categorical_features['seasons'].append(timestamp_features['season'])
        
        # Fit scalers
        if self.normalize_features and all_numerical_features:
            numerical_array = np.array(all_numerical_features)
            self.numerical_scaler.fit(numerical_array)
        
        # Fit encoders
        if all_categorical_features['authors']:
            unique_authors = list(set(all_categorical_features['authors']))[:self.max_authors]
            self.author_encoder.fit(unique_authors + ['<UNK>'])
        
        if all_categorical_features['file_types']:
            unique_file_types = list(set(all_categorical_features['file_types']))
            self.file_type_encoder.fit(unique_file_types + ['<UNK>'])
        
        # Fit file path vectorizer
        all_file_paths = []
        for sample in metadata_samples:
            files = sample.get('files', [])
            paths = [f.get('filename', '') for f in files]
            all_file_paths.extend(paths)
        
        if all_file_paths:
            self.file_path_vectorizer.fit(all_file_paths)
        
        self.is_fitted = True
        print("✅ Metadata processors fitted successfully")
    
    def _get_numerical_features(self, file_features: Dict, author_features: Dict, 
                               timestamp_features: Dict, engineered_features: Dict) -> List[float]:
        """
        Lấy tất cả numerical features thành một vector
        """
        features = []
        
        # File features
        features.extend([
            file_features['num_files'],
            file_features['total_additions'],
            file_features['total_deletions'],
            file_features['total_changes'],
            file_features['num_file_types'],
            file_features['avg_file_depth'],
            file_features['max_file_depth'],
            file_features['min_file_depth'],
            file_features['additions_ratio'],
            file_features['deletions_ratio'],
            file_features['added_files'],
            file_features['modified_files'],
            file_features['deleted_files'],
            file_features['renamed_files'],
            file_features['large_change_files'],
            float(file_features['has_large_changes'])
        ])
        
        # Author features
        features.extend([
            author_features['recent_commits_count'],
            author_features['avg_recent_commit_size'],
            author_features['avg_commit_interval_hours'],
            author_features['commit_frequency_score']
        ])
        
        # Timestamp features
        features.extend([
            timestamp_features['hour_of_day'],
            timestamp_features['day_of_week'],
            timestamp_features['day_of_month'],
            timestamp_features['month'],
            float(timestamp_features['is_weekend']),
            float(timestamp_features['is_business_hours']),
            float(timestamp_features['is_late_night'])
        ])
        
        # Engineered features
        features.extend([
            engineered_features['complexity_score'],
            engineered_features['risk_score'],
            engineered_features['urgency_score'],
            engineered_features['code_churn'],
            float(engineered_features['touches_risky_files']),
            float(engineered_features['touches_config_files']),
            float(engineered_features['touches_doc_files'])
        ])
        
        return features
    
    def process_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """
        Xử lý một sample metadata
        """
        if not self.is_fitted:
            raise ValueError("MetadataProcessor must be fitted before processing samples")
        
        # Extract features
        file_features = self.extract_file_features(sample.get('files', []))
        author_features = self.extract_author_features(
            sample.get('author', {}), 
            sample.get('commit_history', [])
        )
        timestamp_features = self.extract_timestamp_features(sample.get('timestamp', ''))
        engineered_features = self.create_feature_engineering(file_features, author_features, timestamp_features)
        
        result = {}
        
        # Numerical features
        numerical = self._get_numerical_features(file_features, author_features, timestamp_features, engineered_features)
        if self.normalize_features:
            numerical = self.numerical_scaler.transform([numerical])[0]
        result['numerical_features'] = torch.tensor(numerical, dtype=torch.float32)
        
        # Categorical features
        # Author encoding
        author_login = author_features['author_login']
        try:
            author_encoded = self.author_encoder.transform([author_login])[0]
        except ValueError:
            author_encoded = self.author_encoder.transform(['<UNK>'])[0]
        result['author_encoded'] = torch.tensor(author_encoded, dtype=torch.long)
        
        # Season encoding
        season_map = {'spring': 0, 'summer': 1, 'fall': 2, 'winter': 3}
        result['season_encoded'] = torch.tensor(season_map.get(timestamp_features['season'], 0), dtype=torch.long)
          # File types encoding (one-hot or multi-hot)
        try:
            # Get number of classes from fitted encoder
            num_classes = len(self.file_type_encoder.classes_)
        except AttributeError:
            # Fallback if encoder is not fitted or doesn't have classes_ attribute
            num_classes = 10  # Default reasonable size
            
        file_type_vector = np.zeros(num_classes)
        for file_type in file_features['file_types']:
            try:
                idx = self.file_type_encoder.transform([file_type])[0]
                if idx < num_classes:  # Safety check
                    file_type_vector[idx] = 1
            except (ValueError, AttributeError):
                continue
        result['file_types_encoded'] = torch.tensor(file_type_vector, dtype=torch.float32)
        
        return result
    
    def process_batch(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Xử lý một batch samples
        """
        batch_results = {
            'numerical_features': [],
            'author_encoded': [],
            'season_encoded': [],
            'file_types_encoded': []
        }
        
        for sample in samples:
            processed = self.process_sample(sample)
            for key, value in processed.items():
                batch_results[key].append(value)
        
        # Stack tensors
        for key in batch_results:
            batch_results[key] = torch.stack(batch_results[key])
        
        return batch_results
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Trả về dimensions của các feature types
        """
        return {
            'numerical_dim': 33,  # Total numerical features
            'author_vocab_size': len(self.author_encoder.classes_) if hasattr(self.author_encoder, 'classes_') else 1000,
            'season_vocab_size': 4,
            'file_types_dim': len(self.file_type_encoder.classes_) if hasattr(self.file_type_encoder, 'classes_') else 100
        }
