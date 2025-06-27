"""
Module xử lý metadata cho commit.
"""
import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.preprocessing import StandardScaler

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetadataProcessor:
    """Class xử lý metadata cho commit."""
    
    def __init__(self, features_to_include: Optional[List[str]] = None):
        """
        Khởi tạo metadata processor.
        
        Args:
            features_to_include: Danh sách các đặc trưng cần sử dụng, nếu None thì sử dụng tất cả
        """
        self.features_to_include = features_to_include
        self.feature_names = []
        self.numerical_features = []
        self.categorical_features = []
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _identify_feature_types(self, sample_feature: Dict[str, Any]) -> None:
        """
        Xác định loại đặc trưng (số hoặc phân loại).
        
        Args:
            sample_feature: Dict các đặc trưng mẫu
        """
        for feature_name, value in sample_feature.items():
            if self.features_to_include is not None and feature_name not in self.features_to_include:
                continue
                
            self.feature_names.append(feature_name)
            
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.numerical_features.append(feature_name)
            else:
                self.categorical_features.append(feature_name)
    
    def fit(self, features_list: List[Dict[str, Any]]) -> None:
        """
        Fit processor với danh sách đặc trưng.
        
        Args:
            features_list: Danh sách dict đặc trưng
        """
        if not features_list:
            raise ValueError("features_list không được rỗng")
        
        # Xác định loại đặc trưng
        self._identify_feature_types(features_list[0])
        
        # Chuẩn bị dữ liệu để fit scaler
        numerical_data = []
        for features in features_list:
            numerical_values = [features.get(feature, 0) for feature in self.numerical_features]
            numerical_data.append(numerical_values)
        
        # Fit scaler cho đặc trưng số
        if numerical_data:
            self.scaler.fit(numerical_data)
        
        self.is_fitted = True
        logger.info(f"Đã fit metadata processor với {len(self.feature_names)} đặc trưng ({len(self.numerical_features)} số, {len(self.categorical_features)} phân loại)")
    
    def process(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """
        Xử lý danh sách đặc trưng thành mảng numpy.
        
        Args:
            features_list: Danh sách dict đặc trưng
            
        Returns:
            Mảng numpy các đặc trưng đã xử lý
        """
        if not self.is_fitted:
            raise ValueError("Metadata processor chưa được fit. Hãy gọi phương thức fit trước.")
        
        processed_data = []
        
        for features in features_list:
            # Xử lý đặc trưng số
            numerical_values = [features.get(feature, 0) for feature in self.numerical_features]
            
            # Xử lý đặc trưng phân loại
            categorical_values = []
            for feature in self.categorical_features:
                value = features.get(feature, "")
                # Chuyển đổi giá trị phân loại thành số, đơn giản hóa bằng cách chỉ sử dụng 0/1 cho các giá trị bool
                if isinstance(value, bool):
                    categorical_values.append(1 if value else 0)
                else:
                    # Đối với các giá trị khác, chuyển về 0 hoặc hash value
                    try:
                        categorical_values.append(1 if value else 0)
                    except:
                        # Đối với các giá trị phức tạp, sử dụng hash
                        try:
                            categorical_values.append(hash(str(value)) % 100 / 100)  # Scale to 0-1
                        except:
                            categorical_values.append(0)
            
            # Kết hợp các đặc trưng
            feature_vector = numerical_values + categorical_values
            processed_data.append(feature_vector)
        
        # Chuẩn hóa đặc trưng số
        if processed_data and self.numerical_features:
            numerical_data = np.array(processed_data)[:, :len(self.numerical_features)]
            categorical_data = np.array(processed_data)[:, len(self.numerical_features):]
            
            normalized_numerical = self.scaler.transform(numerical_data)
            
            # Kết hợp lại
            if categorical_data.size > 0:
                processed_data = np.hstack((normalized_numerical, categorical_data))
            else:
                processed_data = normalized_numerical
        else:
            processed_data = np.array(processed_data)
        
        return processed_data
    
    def process_single(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Xử lý một dict đặc trưng thành mảng numpy.
        
        Args:
            features: Dict đặc trưng
            
        Returns:
            Mảng numpy các đặc trưng đã xử lý
        """
        return self.process([features])[0]
    
    def save(self, filepath: str) -> None:
        """
        Lưu metadata processor vào file.
        
        Args:
            filepath: Đường dẫn file
        """
        data = {
            'features_to_include': self.features_to_include,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_var': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Đã lưu metadata processor vào {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MetadataProcessor':
        """
        Tải metadata processor từ file.
        
        Args:
            filepath: Đường dẫn file
            
        Returns:
            MetadataProcessor
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processor = cls(features_to_include=data['features_to_include'])
        
        processor.feature_names = data['feature_names']
        processor.numerical_features = data['numerical_features']
        processor.categorical_features = data['categorical_features']
        processor.is_fitted = data['is_fitted']
        
        if data['scaler_mean'] is not None and data['scaler_var'] is not None:
            processor.scaler.mean_ = np.array(data['scaler_mean'])
            processor.scaler.var_ = np.array(data['scaler_var'])
            processor.scaler.scale_ = np.array(data['scaler_scale'])
        
        # Bổ sung gán output_dim để pipeline không lỗi
        processor.output_dim = len(processor.feature_names)
        
        logger.info(f"Đã tải metadata processor từ {filepath}")
        return processor


if __name__ == "__main__":
    # Ví dụ sử dụng
    features_list = [
        {
            'files_changed': 5,
            'additions': 100,
            'deletions': 20,
            'is_merge': False,
            'has_py_files': True
        },
        {
            'files_changed': 2,
            'additions': 50,
            'deletions': 10,
            'is_merge': True,
            'has_py_files': False
        }
    ]
    
    processor = MetadataProcessor()
    processor.fit(features_list)
    
    # Xử lý metadata
    processed = processor.process(features_list)
    print(processed.shape)
    
    # Lưu và tải
    processor.save("metadata_processor.json")
    loaded_processor = MetadataProcessor.load("metadata_processor.json")
