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
import warnings
warnings.filterwarnings('ignore')

class CommitAnalysisSystem:
    def __init__(self):
        """Khởi tạo hệ thống với cấu hình tối ưu"""
        self.vectorizer = TfidfVectorizer(
            max_features=800,
            stop_words='english',
            ngram_range=(1, 1)
        )
        self.model = RandomForestClassifier(
            n_estimators=30,
            max_depth=8,
            n_jobs=1,
            class_weight='balanced'
        )
        self.client = None

    def init_dask_client(self):
        """Khởi tạo Dask client"""
        self.client = Client(n_workers=2, threads_per_worker=1, memory_limit='2GB')

    @staticmethod
    def lightweight_heuristic(msg):
        """Hàm heuristic tĩnh để xử lý song song"""
        if not isinstance(msg, str) or not msg.strip():
            return 0
        msg = msg.lower()[:150]
        return int(any(kw in msg for kw in ['fix', 'bug', 'error', 'fail']))

    def process_large_file(self, input_path, output_dir):
        """Xử lý file lớn với Dask """
        try:
            if self.client:
                self.client.close()
            self.init_dask_client()

            # Đọc file với Dask
            ddf = dd.read_csv(
                str(input_path),
                blocksize="20MB",
                dtype={'message': 'string'},
                usecols=['commit', 'message'],
                na_values=['', 'NA', 'N/A', 'nan']
            )
            
            # Sửa lỗi: Thay .notna() bằng .notnull() cho Dask
            ddf = ddf[ddf['message'].notnull()]
            
            # Gán nhãn
            ddf['is_critical'] = ddf['message'].map(
                self.lightweight_heuristic,
                meta=('is_critical', 'int8')
            )
            
           # Lưu kết quả (đã sửa phần compute)
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Sửa lỗi: Gọi compute() trực tiếp trên to_csv()
            ddf.to_csv(
                str(output_dir / "part_*.csv"),
                index=False
            )
            
            return True
        except Exception as e:
            print(f"🚨 Lỗi xử lý file: {str(e)}")
            return False
        finally:
            if self.client:
                self.client.close()

    def clean_data(self, df):
        """Làm sạch dữ liệu"""
        if 'message' not in df.columns:
            raise ValueError("Thiếu cột 'message' trong dữ liệu")
        df['message'] = df['message'].astype('string').fillna('')
        return df[df['message'].str.strip() != '']

    def auto_label(self, df):
        """Gán nhãn tự động"""
        df = self.clean_data(df)
        df['is_critical'] = df['message'].apply(self.lightweight_heuristic)
        return df

    def train_model(self, df):
        """Huấn luyện mô hình"""
        X = self.vectorizer.fit_transform(df['message'])
        y = df['is_critical'].values
        self.model.fit(X, y)

    def evaluate(self, test_df):
        """Đánh giá mô hình"""
        X_test = self.vectorizer.transform(test_df['message'])
        y_test = test_df['is_critical'].values
        print(classification_report(y_test, self.model.predict(X_test)))

    def save_model(self, path):
        """Lưu mô hình"""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer
        }, str(path))

def main():
    print("🚀 Bắt đầu phân tích commit...")
    system = CommitAnalysisSystem()
    
    input_path = Path("D:/Project/KLTN04/data/oneline.csv")
    output_dir = Path("D:/Project/KLTN04/data/processed")
    
    if system.process_large_file(input_path, output_dir):
        print("✅ Đã xử lý file thành công")
        
        # Nạp và xử lý dữ liệu
        df = pd.concat([pd.read_csv(f) for f in output_dir.glob("part_*.csv")])
        df = system.auto_label(df)
        
        # Huấn luyện và đánh giá
        system.train_model(df)
        test_df = df.sample(frac=0.2, random_state=42)
        system.evaluate(test_df)
        
        # Lưu mô hình
        model_path = "backend/models/commit_classifier.joblib"
        system.save_model(model_path)
        print(f"💾 Đã lưu mô hình tại: {model_path}")

if __name__ == "__main__":
    main()