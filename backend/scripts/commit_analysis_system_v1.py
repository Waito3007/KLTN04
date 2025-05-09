# File: backend/scripts/commit_analysis_system.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import dask.dataframe as dd
import joblib
from dask.distributed import Client
import logging
from typing import Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CommitAnalysisSystem:
    """Hệ thống phân tích commit tự động với khả năng xử lý dữ liệu lớn"""
    
    VERSION = "1.0.0"
    
    def __init__(self, model_params: Optional[dict] = None, 
                 vectorizer_params: Optional[dict] = None):
        """
        Khởi tạo hệ thống phân tích commit
        
        Args:
            model_params: Tham số cho RandomForestClassifier
            vectorizer_params: Tham số cho TfidfVectorizer
        """
        # Cấu hình mặc định
        default_vectorizer_params = {
            'max_features': 1000,
            'stop_words': 'english',
            'ngram_range': (1, 2),  # Thêm bigram
            'min_df': 5,
            'max_df': 0.8
        }
        
        default_model_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        self.vectorizer = TfidfVectorizer(**(vectorizer_params or default_vectorizer_params))
        self.model = RandomForestClassifier(**(model_params or default_model_params))
        self.client = None
        self._is_trained = False

    def init_dask_client(self, **kwargs):
        """Khởi tạo Dask client với cấu hình tùy chọn"""
        default_config = {
            'n_workers': 2,
            'threads_per_worker': 1,
            'memory_limit': '2GB',
            'silence_logs': logging.ERROR
        }
        config = {**default_config, **kwargs}
        
        try:
            self.client = Client(**config)
            logger.info(f"Dask client initialized with config: {config}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Dask client: {str(e)}")
            return False

    @staticmethod
    def lightweight_heuristic(msg: str) -> int:
        """Phân loại commit sử dụng heuristic đơn giản
        
        Args:
            msg: Nội dung commit message
            
        Returns:
            1 nếu là commit quan trọng (bugfix), 0 nếu không
        """
        if not isinstance(msg, str) or not msg.strip():
            return 0
            
        msg = msg.lower()[:200]  # Giới hạn độ dài xử lý
        keywords = {
            'fix', 'bug', 'error', 'fail', 'patch', 
            'resolve', 'crash', 'defect', 'issue'
        }
        return int(any(kw in msg for kw in keywords))

    def process_large_file(self, input_path: Union[str, Path], output_dir: Union[str, Path]) -> bool:
        """Xử lý file dữ liệu lớn bằng Dask"""
        try:
            input_path = Path(input_path)
            output_dir = Path(output_dir)

            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                return False

            logger.info(f"Starting processing large file: {input_path}")
            start_time = datetime.now()

            # Khởi tạo Dask client
            if not self.init_dask_client():
                return False

            try:
                # Đọc và xử lý dữ liệu
                ddf = dd.read_csv(
                    str(input_path),
                    blocksize="10MB",  # Giảm kích thước block để an toàn
                    dtype={'message': 'string'},
                    usecols=['commit', 'message'],
                    na_values=['', 'NA', 'N/A', 'nan']
                )

                # Lọc và gán nhãn
                ddf = ddf[ddf['message'].notnull()]
                ddf['is_critical'] = ddf['message'].map(
                    self.lightweight_heuristic,
                    meta=('is_critical', 'int8')
                )

                # Lưu kết quả
                output_dir.mkdir(exist_ok=True, parents=True)
                output_path = str(output_dir / f"processed_{input_path.stem}.csv")

                # Sử dụng dask.dataframe.to_csv với single_file=True
                ddf.to_csv(
                    output_path,
                    index=False,
                    single_file=True
                )

                logger.info(f"Processing completed in {datetime.now() - start_time}")
                logger.info(f"Results saved to: {output_path}")
                return True

            except Exception as e:
                logger.exception(f"Error during processing: {str(e)}")
                return False

        except Exception as e:
            logger.exception(f"System error: {str(e)}")
            return False

        finally:
            if self.client:
                self.client.close()
                self.client = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch dữ liệu đầu vào
        
        Args:
            df: DataFrame chứa dữ liệu commit
            
        Returns:
            DataFrame đã được làm sạch
        """
        if 'message' not in df.columns:
            raise ValueError("Input data must contain 'message' column")
            
        df = df.copy()
        df['message'] = df['message'].astype('string').fillna('')
        return df[df['message'].str.strip() != '']

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tự động gán nhãn cho dữ liệu commit
        
        Args:
            df: DataFrame chứa các commit message
            
        Returns:
            DataFrame đã được gán nhãn
        """
        try:
            df = self.clean_data(df)
            df['is_critical'] = df['message'].apply(self.lightweight_heuristic)
            logger.info(f"Label distribution:\n{df['is_critical'].value_counts()}")
            return df
        except Exception as e:
            logger.error(f"Auto-labeling failed: {str(e)}")
            raise

    def train_model(self, df: pd.DataFrame) -> bool:
        """Huấn luyện mô hình phân loại commit
        
        Args:
            df: DataFrame đã được gán nhãn
            
        Returns:
            True nếu huấn luyện thành công
        """
        try:
            logger.info("Starting model training...")
            
            X = self.vectorizer.fit_transform(df['message'])
            y = df['is_critical'].values
            
            self.model.fit(X, y)
            self._is_trained = True
            
            logger.info("Model training completed successfully")
            return True
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

    def evaluate(self, test_df: pd.DataFrame) -> None:
        """Đánh giá hiệu suất mô hình
        
        Args:
            test_df: DataFrame chứa dữ liệu test
        """
        if not self._is_trained:
            logger.warning("Model has not been trained yet")
            return
            
        X_test = self.vectorizer.transform(test_df['message'])
        y_test = test_df['is_critical'].values
        y_pred = self.model.predict(X_test)
        
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=['normal', 'critical']
        )
        logger.info(f"\nModel evaluation:\n{report}")

    def save_model(self, path: Union[str, Path]) -> bool:
        """Lưu mô hình và vectorizer
        
        Args:
            path: Đường dẫn lưu model
            
        Returns:
            True nếu lưu thành công
        """
        try:
            path = Path(path)
            path.parent.mkdir(exist_ok=True, parents=True)
            
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'version': self.VERSION,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, str(path))
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False

    @classmethod
    def load_model(cls, path: Union[str, Path]):
        """Tải mô hình đã lưu
        
        Args:
            path: Đường dẫn đến file model
            
        Returns:
            Instance của CommitAnalysisSystem với model đã tải
        """
        try:
            path = Path(path)
            model_data = joblib.load(str(path))
            
            system = cls()
            system.model = model_data['model']
            system.vectorizer = model_data['vectorizer']
            system._is_trained = True
            
            logger.info(f"Loaded model (v{model_data.get('version', 'unknown')} "
                       f"created at {model_data.get('timestamp', 'unknown')}")
            return system
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

def main():
    """Entry point cho ứng dụng"""
    try:
        logger.info("🚀 Starting commit analysis system")
        
        # Cấu hình đường dẫn
        input_path = Path("D:/Project/KLTN04/data/oneline.csv")
        output_dir = Path("D:/Project/KLTN04/data/processed")
        model_path = Path("backend/models/commit_classifier_v1.joblib")
        
        # Khởi tạo hệ thống
        system = CommitAnalysisSystem()
        
        # Xử lý dữ liệu lớn
        if system.process_large_file(input_path, output_dir):
            # Tổng hợp kết quả
            df = pd.concat([
                pd.read_csv(f) 
                for f in output_dir.glob("processed_*.csv")
            ])
            
            # Gán nhãn và huấn luyện
            labeled_data = system.auto_label(df)
            system.train_model(labeled_data)
            
            # Đánh giá trên tập test
            test_df = labeled_data.sample(frac=0.2, random_state=42)
            system.evaluate(test_df)
            
            # Lưu model
            if system.save_model(model_path):
                logger.info(f"✅ Pipeline completed successfully. Model saved to {model_path}")
        
    except Exception as e:
        logger.exception("❌ Critical error in main pipeline")
    finally:
        logger.info("🏁 System shutdown")

if __name__ == "__main__":
    main()