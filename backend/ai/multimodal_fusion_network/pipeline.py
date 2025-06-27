"""
Module tích hợp các thành phần của pipeline để phân tích commit.
"""
import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from datetime import datetime


from data_collection.github_collector import GitHubDataCollector
from data_processing.commit_processor import CommitDataProcessor
from data_processing.text_processor import TextProcessor
from data_processing.metadata_processor import MetadataProcessor
from data_processing.dataset import create_data_loaders
from models.multimodal_fusion_model import EnhancedMultimodalFusionModel, create_model_config
from models.predictor import CommitPredictor
from training.trainer import train_multimodal_fusion_model
from evaluation.evaluator import evaluate_multimodal_fusion_model

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommitAnalysisPipeline:
    """Class tích hợp toàn bộ pipeline phân tích commit."""
    
    def __init__(
        self,
        base_dir: str,
        github_token: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Khởi tạo pipeline.
        
        Args:
            base_dir: Thư mục cơ sở cho lưu trữ dữ liệu và mô hình
            github_token: Token truy cập GitHub API (nếu cần thu thập dữ liệu)
            device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
        """
        self.base_dir = base_dir
        self.github_token = github_token
        
        # Xác định thiết bị
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Tạo các thư mục cần thiết
        self.data_dir = os.path.join(base_dir, 'data')
        self.processed_dir = os.path.join(base_dir, 'processed')
        self.models_dir = os.path.join(base_dir, 'models')
        self.checkpoints_dir = os.path.join(base_dir, 'checkpoints')
        self.results_dir = os.path.join(base_dir, 'results')
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Khởi tạo các components
        self.github_collector = None
        self.commit_processor = None
        self.text_processor = None
        self.metadata_processor = None
        self.model = None
        self.predictor = None
    
    def collect_data(
        self,
        repo_names: List[str],
        max_commits_per_repo: int = 1000,
        output_file: Optional[str] = None
    ) -> str:
        """
        Thu thập dữ liệu commit từ GitHub.
        
        Args:
            repo_names: Danh sách tên repository (dạng 'owner/repo')
            max_commits_per_repo: Số lượng commit tối đa cho mỗi repo
            output_file: Tên file đầu ra (nếu None thì tạo tự động)
            
        Returns:
            Đường dẫn đến file dữ liệu
        """
        if self.github_token is None:
            raise ValueError("Cần cung cấp GitHub token để thu thập dữ liệu")
        
        # Tạo collector nếu chưa có
        if self.github_collector is None:
            self.github_collector = GitHubCommitCollector(self.github_token)
        
        # Tạo tên file nếu chưa có
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.data_dir, f"github_commits_{timestamp}.json")
        
        # Thu thập dữ liệu
        logger.info(f"Bắt đầu thu thập dữ liệu từ {len(repo_names)} repositories")
        
        all_commits = []
        for repo_name in repo_names:
            logger.info(f"Đang thu thập từ {repo_name}")
            repo_commits = self.github_collector.collect_commits(repo_name, max_commits=max_commits_per_repo)
            all_commits.extend(repo_commits)
            logger.info(f"Đã thu thập {len(repo_commits)} commits từ {repo_name}")
        
        # Lưu dữ liệu
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'collected_at': datetime.now().isoformat(),
                    'repositories': repo_names,
                    'total_commits': len(all_commits)
                },
                'commits': all_commits
            }, f, indent=2)
        
        logger.info(f"Đã thu thập và lưu {len(all_commits)} commits vào {output_file}")
        
        return output_file
    
    def process_data(
        self,
        input_file: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        auto_labeling: bool = True,
        random_seed: int = 42
    ) -> Tuple[str, str, str]:
        """
        Xử lý dữ liệu commit thô và tạo tập train/val/test.
        
        Args:
            input_file: Đường dẫn đến file dữ liệu thô
            train_ratio: Tỷ lệ dữ liệu cho tập train
            val_ratio: Tỷ lệ dữ liệu cho tập validation
            test_ratio: Tỷ lệ dữ liệu cho tập test
            auto_labeling: Có tự động gán nhãn hay không
            random_seed: Seed ngẫu nhiên
            
        Returns:
            Tuple (train_path, val_path, test_path)
        """
        # Tạo processor nếu chưa có
        if self.commit_processor is None:
            self.commit_processor = CommitDataProcessor(input_file)
        
        # Xử lý dữ liệu
        logger.info(f"Bắt đầu xử lý dữ liệu từ {input_file}")
        
        processed_data = self.commit_processor.process_commits(auto_labeling=auto_labeling)
        
        # Trích xuất các đặc trưng text và metadata
        texts = [item['text'] for item in processed_data]
        features = [item['features'] for item in processed_data]
        
        # Tạo và fit text processor
        self.text_processor = TextProcessor()
        self.text_processor.fit(texts)
        
        # Tạo và fit metadata processor
        self.metadata_processor = MetadataProcessor()
        self.metadata_processor.fit(features)
        
        # Lưu processors
        text_processor_path = os.path.join(self.processed_dir, 'text_processor.json')
        metadata_processor_path = os.path.join(self.processed_dir, 'metadata_processor.json')
        
        self.text_processor.save(text_processor_path)
        self.metadata_processor.save(metadata_processor_path)
        
        logger.info(f"Đã lưu text processor với vocab_size={self.text_processor.vocab_size}")
        logger.info(f"Đã lưu metadata processor với {len(self.metadata_processor.feature_names)} features")
        
        # Chia tập dữ liệu
        np.random.seed(random_seed)
        indices = np.random.permutation(len(processed_data))
        
        train_size = int(len(processed_data) * train_ratio)
        val_size = int(len(processed_data) * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = [processed_data[i] for i in train_indices]
        val_data = [processed_data[i] for i in val_indices]
        test_data = [processed_data[i] for i in test_indices]
        
        # Lưu các tập dữ liệu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_path = os.path.join(self.processed_dir, f"train_{timestamp}.json")
        val_path = os.path.join(self.processed_dir, f"val_{timestamp}.json")
        test_path = os.path.join(self.processed_dir, f"test_{timestamp}.json")
        
        # Lưu metadata về quá trình xử lý
        metadata = {
            'processed_at': datetime.now().isoformat(),
            'input_file': input_file,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'auto_labeling': auto_labeling,
            'random_seed': random_seed,
            'text_processor': text_processor_path,
            'metadata_processor': metadata_processor_path
        }
        
        # Lưu các tập dữ liệu
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump({'metadata': metadata, 'data': train_data}, f, indent=2)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump({'metadata': metadata, 'data': val_data}, f, indent=2)
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump({'metadata': metadata, 'data': test_data}, f, indent=2)
        
        logger.info(f"Đã chia và lưu dữ liệu: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_path, val_path, test_path
    
    def train_model(
        self,
        train_path: str,
        val_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 5,
        text_encoder_method: str = 'transformer'
    ) -> Tuple[EnhancedMultimodalFusionModel, str]:
        if self.text_processor is None:
            # Sửa đường dẫn để load đúng file processor đã fit
            processor_path = os.path.join('data', 'processed', 'text_processor.json')
            self.text_processor = TextProcessor.load(processor_path)
            if self.text_processor is None or not self.text_processor.is_fitted:
                raise RuntimeError("TextProcessor chưa được khởi tạo hoặc chưa fit. Hãy fit và load TextProcessor trước khi train.")
        if self.metadata_processor is None:
            meta_processor_path = os.path.join('data', 'processed', 'metadata_processor.json')
            self.metadata_processor = MetadataProcessor.load(meta_processor_path)
            if self.metadata_processor is None or not hasattr(self.metadata_processor, 'output_dim'):
                raise RuntimeError("MetadataProcessor chưa được khởi tạo hoặc chưa fit. Hãy fit và load MetadataProcessor trước khi train.")
        """
        Huấn luyện mô hình fusion đa phương thức.
        
        Args:
            train_path: Đường dẫn đến file dữ liệu train
            val_path: Đường dẫn đến file dữ liệu validation
            model_config: Cấu hình mô hình (nếu None thì tạo tự động)
            batch_size: Kích thước batch
            num_epochs: Số epochs
            learning_rate: Learning rate
            early_stopping_patience: Số epochs chờ trước khi dừng sớm
            text_encoder_method: 'lstm' hoặc 'transformer'
            
        Returns:
            Tuple (model, checkpoint_path)
        """
        # Tạo data loaders
        train_loader, val_loader, _ = create_data_loaders(
            train_path=train_path,
            val_path=val_path,
            test_path=val_path,  # Không sử dụng test_loader trong quá trình huấn luyện
            text_processor=self.text_processor,
            metadata_processor=self.metadata_processor,
            batch_size=batch_size
        )
        
        # Tạo cấu hình mô hình nếu chưa có
        if model_config is None:
            # Xác định các task heads từ dữ liệu
            with open(train_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            sample = train_data['data'][0]
            task_heads = {}
            
            for task_name, value in sample['labels'].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if value % 1 == 0:  # Nếu là số nguyên
                        # Đếm số lớp
                        all_values = [item['labels'].get(task_name, 0) for item in train_data['data']]
                        num_classes = len(set(all_values))
                        task_heads[task_name] = {'num_classes': num_classes}
                    else:  # Nếu là số thực
                        task_heads[task_name] = {'type': 'regression'}
            
            model_config = create_model_config(
                vocab_size=self.text_processor.vocab_size,
                metadata_dim=self.metadata_processor.output_dim,
                task_heads=task_heads,
                text_encoder_method=text_encoder_method,
                fusion_dim=256
            )
        
        # Tạo thư mục lưu checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(self.checkpoints_dir, f"training_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Lưu cấu hình mô hình
        with open(os.path.join(checkpoint_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
            # Chuyển đổi các đối tượng không serializable sang string
            config_serializable = {}
            for k, v in model_config.items():
                if isinstance(v, dict):
                    config_serializable[k] = {}
                    for k2, v2 in v.items():
                        config_serializable[k][k2] = str(v2) if not isinstance(v2, (dict, list, int, float, str, bool, type(None))) else v2
                else:
                    config_serializable[k] = str(v) if not isinstance(v, (dict, list, int, float, str, bool, type(None))) else v
            
            json.dump(config_serializable, f, indent=2)
        
        # Huấn luyện mô hình
        logger.info(f"Bắt đầu huấn luyện mô hình với {num_epochs} epochs")
        
        model, history = train_multimodal_fusion_model(
            model_config=model_config,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=1e-5,
            task_weights=None,  # Sử dụng trọng số mặc định
            checkpoint_dir=checkpoint_dir,
            device=self.device,
            early_stopping_patience=early_stopping_patience,
            log_interval=1
        )
        
        self.model = model
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        logger.info(f"Đã hoàn thành huấn luyện và lưu mô hình tại {best_checkpoint_path}")
        
        return model, best_checkpoint_path
    
    def evaluate_model(
        self,
        test_path: str,
        model_path: Optional[str] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Đánh giá mô hình trên tập test.
        
        Args:
            test_path: Đường dẫn đến file dữ liệu test
            model_path: Đường dẫn đến checkpoint của mô hình (nếu None thì sử dụng mô hình hiện tại)
            batch_size: Kích thước batch
            
        Returns:
            Dict kết quả đánh giá
        """
        # Nếu cần tải mô hình từ checkpoint
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint['model_config']
            
            self.model = EnhancedMultimodalFusionModel(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
        
        if self.model is None:
            raise ValueError("Chưa có mô hình nào được huấn luyện hoặc tải")
        
        # Tạo test loader
        _, _, test_loader = create_data_loaders(
            train_path=test_path,  # Không sử dụng trong đánh giá
            val_path=test_path,    # Không sử dụng trong đánh giá
            test_path=test_path,
            text_processor=self.text_processor,
            metadata_processor=self.metadata_processor,
            batch_size=batch_size
        )
        
        # Tạo thư mục lưu kết quả
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.results_dir, f"evaluation_{timestamp}")
        
        # Đánh giá mô hình
        logger.info(f"Bắt đầu đánh giá mô hình trên tập test")
        
        results = evaluate_multimodal_fusion_model(
            model=self.model,
            test_loader=test_loader,
            output_dir=results_dir,
            device=self.device
        )
        
        logger.info(f"Đã hoàn thành đánh giá và lưu kết quả tại {results_dir}")
        
        return results
    
    def load_predictor(self, model_path: str) -> CommitPredictor:
        """
        Tải predictor từ checkpoint của mô hình.
        
        Args:
            model_path: Đường dẫn đến checkpoint của mô hình
            
        Returns:
            CommitPredictor
        """
        # Đảm bảo có text_processor và metadata_processor
        if self.text_processor is None or self.metadata_processor is None:
            # Tìm đường dẫn đến processors từ checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'metadata' in checkpoint and 'text_processor' in checkpoint['metadata'] and 'metadata_processor' in checkpoint['metadata']:
                text_processor_path = checkpoint['metadata']['text_processor']
                metadata_processor_path = checkpoint['metadata']['metadata_processor']
            else:
                # Sử dụng processors mặc định trong thư mục processed
                text_processor_path = os.path.join(self.processed_dir, 'text_processor.json')
                metadata_processor_path = os.path.join(self.processed_dir, 'metadata_processor.json')
            
            self.text_processor = TextProcessor.load(text_processor_path)
            self.metadata_processor = MetadataProcessor.load(metadata_processor_path)
        
        # Tạo predictor
        self.predictor = CommitPredictor(
            model_path=model_path,
            text_processor_path=os.path.join(self.processed_dir, 'text_processor.json'),
            metadata_processor_path=os.path.join(self.processed_dir, 'metadata_processor.json'),
            device=self.device
        )
        
        logger.info(f"Đã tải predictor từ {model_path}")
        
        return self.predictor
    
    def predict(self, commit_message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dự đoán cho một commit message.
        
        Args:
            commit_message: Nội dung commit message
            metadata: Dict metadata (nếu None thì tạo metadata rỗng)
            
        Returns:
            Dict kết quả dự đoán và đề xuất
        """
        if self.predictor is None:
            raise ValueError("Chưa tải predictor. Gọi phương thức load_predictor trước.")
        
        # Dự đoán
        prediction = self.predictor.predict(commit_message, metadata)
        
        # Tạo đề xuất
        recommendations = self.predictor.generate_recommendations(prediction)
        
        return {
            'prediction': prediction,
            'recommendations': recommendations
        }
    
    def batch_predict(self, commit_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dự đoán cho một batch commit.
        
        Args:
            commit_data: List các Dict chứa thông tin commit
            
        Returns:
            List các Dict kết quả dự đoán và đề xuất
        """
        if self.predictor is None:
            raise ValueError("Chưa tải predictor. Gọi phương thức load_predictor trước.")
        
        # Dự đoán batch
        results = self.predictor.batch_predict(commit_data)
        
        # Tạo đề xuất cho từng kết quả
        for result in results:
            result['recommendations'] = self.predictor.generate_recommendations(result['prediction'])
        
        return results


def run_end_to_end_pipeline(
    base_dir: str,
    github_token: str,
    repo_names: List[str],
    max_commits_per_repo: int = 1000,
    batch_size: int = 32,
    num_epochs: int = 50,
    text_encoder_method: str = 'transformer'
) -> CommitAnalysisPipeline:
    """
    Chạy toàn bộ pipeline end-to-end từ thu thập dữ liệu đến đánh giá mô hình.
    
    Args:
        base_dir: Thư mục cơ sở cho lưu trữ dữ liệu và mô hình
        github_token: Token truy cập GitHub API
        repo_names: Danh sách tên repository (dạng 'owner/repo')
        max_commits_per_repo: Số lượng commit tối đa cho mỗi repo
        batch_size: Kích thước batch
        num_epochs: Số epochs
        text_encoder_method: 'lstm' hoặc 'transformer'
        
    Returns:
        CommitAnalysisPipeline đã được huấn luyện
    """
    # Khởi tạo pipeline
    pipeline = CommitAnalysisPipeline(base_dir, github_token)
    
    # Thu thập dữ liệu
    data_file = pipeline.collect_data(repo_names, max_commits_per_repo)
    
    # Xử lý dữ liệu
    train_path, val_path, test_path = pipeline.process_data(data_file)
    
    # Huấn luyện mô hình
    model, checkpoint_path = pipeline.train_model(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        text_encoder_method=text_encoder_method
    )
    
    # Đánh giá mô hình
    eval_results = pipeline.evaluate_model(test_path, checkpoint_path, batch_size)
    
    # Tải predictor
    pipeline.load_predictor(checkpoint_path)
    
    return pipeline


if __name__ == "__main__":
    ()
    # Ví dụ sử dụng
    # Để chạy end-to-end pipeline
    # pipeline = run_end_to_end_pipeline(
    #     base_dir="commit_analysis",
    #     github_token="your_github_token",
    #     repo_names=["owner1/repo1", "owner2/repo2"],
    #     max_commits_per_repo=1000,
    #     batch_size=32,
    #     num_epochs=50,
    #     text_encoder_method='transformer'
    # )
    
    # Hoặc để sử dụng từng bước
    # pipeline = CommitAnalysisPipeline("commit_analysis", "your_github_token")
    # data_file = pipeline.collect_data(["owner/repo"], 1000)
    # train_path, val_path, test_path = pipeline.process_data(data_file)
    # model, checkpoint_path = pipeline.train_model(train_path, val_path)
    # eval_results = pipeline.evaluate_model(test_path, checkpoint_path)
    # pipeline.load_predictor(checkpoint_path)
    # 
    # # Dự đoán cho một commit
    # prediction = pipeline.predict(
    #     commit_message="fix: resolve authentication bug in login module",
    #     metadata={
    #         "author": "developer1",
    #         "files_changed": 3,
    #         "additions": 25,
    #         "deletions": 10
    #     }
    # )
    # 
    # print(prediction)
