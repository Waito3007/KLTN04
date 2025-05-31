import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset
from ai.models.hierarchical_attention import HierarchicalAttentionNetwork
from ai.data_preprocessing.text_processor import TextProcessor
from ai.data_preprocessing.embedding_loader import EmbeddingLoader
from ai.training.multitask_trainer import MultiTaskTrainer
# from ai.training.loss_functions import UncertaintyWeightingLoss # Không sử dụng trực tiếp, MultiTaskTrainer có thể tự quản lý hoặc dùng loss_fns
from ai.evaluation.metrics_calculator import calc_metrics
import numpy as np
import json
import glob # Mặc dù import nhưng không thấy sử dụng trong file này
import time # Mặc dù import nhưng không thấy sử dụng trong file này
from datetime import datetime
import random # Thêm để xáo trộn dữ liệu

# Các class và hàm định nghĩa ở global scope (như trong file gốc của bạn)
def save_training_log(epoch, train_loss, val_loss, task_metrics, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Epoch {epoch} ({timestamp}) ===\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Val Loss: {val_loss:.4f}\n")
        for task, metrics_dict in task_metrics.items(): # Sửa tên biến để rõ ràng hơn
            metrics_str = ", ".join([f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics_dict.items()])
            f.write(f"{task}: {{{metrics_str}}}\n")
        f.write("-" * 50 + "\n")

class CommitDataset(Dataset):
    def __init__(self, samples, processor, embed_loader, author_map=None, repo_map=None):
        self.samples = samples
        self.processor = processor
        self.embed_loader = embed_loader
        self.author_map = author_map if author_map is not None else {}
        self.repo_map = repo_map if repo_map is not None else {}
        self.embedding_cache = {}
        
        # Chuẩn hóa ánh xạ nhãn cho từng task
        self.purpose_map = {
            'Feature Implementation': 0,
            'Bug Fix': 1,
            'Refactoring': 2,
            'Documentation Update': 3,
            'Test Update': 4,
            'Security Patch': 5,
            'Code Style/Formatting': 6,
            'Build/CI/CD Script Update': 7,
            'Other': 8
        }
        self.sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.tech_vocab = [
            'python', 'fastapi', 'react', 'javascript', 'typescript', 'docker', 'sqlalchemy', 'pytorch', 'spacy',
            'css', 'html', 'postgresql', 'mysql', 'mongodb', 'redis', 'vue', 'angular', 'flask', 'django',
            'node', 'express', 'graphql', 'rest', 'api', 'gitlab', 'github', 'ci', 'cd', 'kubernetes', 'helm',
            'pytest', 'unittest', 'junit', 'cicd', 'github actions', 'travis', 'jenkins', 'circleci', 'webpack',
            'babel', 'vite', 'npm', 'yarn', 'pip', 'poetry', 'black', 'flake8', 'isort', 'prettier', 'eslint',
            'jwt', 'oauth', 'sso', 'celery', 'rabbitmq', 'kafka', 'grpc', 'protobuf', 'swagger', 'openapi',
            'sentry', 'prometheus', 'grafana', 'nginx', 'apache', 'linux', 'ubuntu', 'windows', 'macos',
            'aws', 'azure', 'gcp', 'firebase', 'heroku', 'netlify', 'vercel', 'tailwind', 'bootstrap', 'material ui'
        ]
        
        # Add mapping for commit types
        self.commit_type_map = {
            'feat': 0,
            'fix': 1,
            'docs': 2,
            'refactor': 3,
            'style': 4,
            'test': 5,
            'chore': 6,
            'uncategorized': 7
        }

        # Track user-specific commit statistics (Phần này có thể không cần thiết cho Dataset)
        # self.user_commit_stats = {}
        # for sample in self.samples:
        #     user = sample.get('source_info', {}).get('author_name', 'Unknown')
        #     commit_type = sample.get('commit_type', 'uncategorized') # Cần đảm bảo field 'commit_type' có trong sample
        #     if user not in self.user_commit_stats:
        #         self.user_commit_stats[user] = {ctype: 0 for ctype in self.commit_type_map.keys()}
        #     if commit_type in self.commit_type_map:
        #         self.user_commit_stats[user][commit_type] += 1

        # Pre-compute embeddings for all words in the dataset
        print("Pre-computing word embeddings...")
        self._precompute_embeddings()
        
    def _precompute_embeddings(self):
        # Collect all unique words
        unique_words = set()
        for sample in self.samples:
            # Giả định 'raw_text' tồn tại và processor.process_document trả về list các list từ
            doc_tokens = self.processor.process_document(sample.get('raw_text', '')) 
            for sent_tokens in doc_tokens:
                unique_words.update(str(word) for word in sent_tokens if str(word)) # Chỉ add nếu từ không rỗng

        # Pre-compute embeddings
        for word in unique_words:
            if word not in self.embedding_cache: # Chỉ tính nếu từ chưa có trong cache
                self.embedding_cache[word] = self.embed_loader.get_word_embedding(word)
        print(f"Cached embeddings for {len(self.embedding_cache)} unique words")

    def _detect_commit_type(self, raw_text):
        # Detect commit type from raw text based on conventional commit format
        if ':' not in raw_text:
            return 'uncategorized'
        prefix = raw_text.split(':')[0].lower().strip()
        if prefix in self.commit_type_map:
            return prefix
        # Các quy tắc heuristic
        if 'feat' in prefix or 'feature' in prefix: return 'feat'
        if 'fix' in prefix or 'bug' in prefix: return 'fix'
        if 'doc' in prefix: return 'docs'
        if 'refactor' in prefix: return 'refactor'
        if 'style' in prefix or 'format' in prefix: return 'style'
        if 'test' in prefix: return 'test'
        if 'chore' in prefix or 'build' in prefix or 'ci' in prefix: return 'chore'
        return 'uncategorized'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_text = sample.get('raw_text', '')
        doc_tokens = self.processor.process_document(raw_text) # list of lists of words/tokens
          # Khởi tạo embed_doc
        # Kích thước embedding từ model config
        if self.embed_loader.embedding_type == 'codebert':
            embed_dim = self.embed_loader.model.config.hidden_size
        else:
            embed_dim = 768  # Giá trị mặc định
        
        embed_doc = np.zeros((self.processor.max_sent_len, self.processor.max_word_len, embed_dim), dtype=np.float32)

        # Use cached embeddings
        for i, sent_tokens in enumerate(doc_tokens):
            if i >= self.processor.max_sent_len: break
            for j, word_token in enumerate(sent_tokens):
                if j >= self.processor.max_word_len: break
                word_str = str(word_token)
                if word_str in self.embedding_cache:
                    embed_doc[i, j] = self.embedding_cache[word_str]
                # else: xử lý từ không có trong cache (ví dụ: vector zero hoặc <unk> vector)
        
        embed_doc_tensor = torch.tensor(embed_doc, dtype=torch.float32)
        
        commit_type_str = sample.get('commit_type') # Lấy từ dữ liệu đã gán nhãn trước
        if not commit_type_str: # Nếu chưa có, thử detect
            commit_type_str = self._detect_commit_type(raw_text)
        commit_type_idx = self.commit_type_map.get(commit_type_str, self.commit_type_map['uncategorized'])

        purpose = self.purpose_map.get(sample.get('purpose', 'Other'), self.purpose_map['Other'])
        suspicious = int(sample.get('suspicious', 0))
        
        tech_tags = sample.get('tech_tags', [])
        tech_idx = 0 # Mặc định nếu không có tech_tag hợp lệ
        if isinstance(tech_tags, list) and tech_tags:
            first_valid_tag = next((tag for tag in tech_tags if tag in self.tech_vocab), None)
            if first_valid_tag:
                tech_idx = self.tech_vocab.index(first_valid_tag)
        
        sentiment = self.sentiment_map.get(sample.get('sentiment', 'neutral'), self.sentiment_map['neutral'])
        
        source_info = sample.get('source_info', {})
        author_idx = self.author_map.get(source_info.get('author_name', 'Unknown'), 0) # Cần giá trị mặc định nếu 'Unknown' không có trong map
        repo_idx = self.repo_map.get(source_info.get('repo_id', 'Unknown'), 0) # Tương tự
        
        labels_tensor_dict = {
            'commit_type': torch.tensor(commit_type_idx, dtype=torch.long),
            'purpose': torch.tensor(purpose, dtype=torch.long),
            'suspicious': torch.tensor(suspicious, dtype=torch.long),
            'tech_tag': torch.tensor(tech_idx, dtype=torch.long),
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'author': torch.tensor(author_idx, dtype=torch.long),
            'source_repo': torch.tensor(repo_idx, dtype=torch.long)
        }
        
        return {
            'input': embed_doc_tensor,
            'labels': labels_tensor_dict
        }

def load_commit_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # Giả sử file JSON chứa một list các sample, hoặc một dict có key 'samples'
            content = json.load(f)
            if isinstance(content, list):
                samples = content
            elif isinstance(content, dict) and 'samples' in content:
                samples = content['samples']
            else:
                raise ValueError("Định dạng JSON không hợp lệ. Cần một list các sample hoặc dict có key 'samples'.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu {json_path}. Tạo dữ liệu giả.")
        samples = [ # Tạo vài mẫu giả để chương trình không crash
            {"raw_text": "feat: example feature", "purpose": "Feature Implementation", "suspicious": 0, "tech_tags": ["python"], "sentiment": "positive", "source_info": {"author_name": "test_author", "repo_id": "test_repo"}, "commit_type": "feat"},
            {"raw_text": "fix: example bug fix", "purpose": "Bug Fix", "suspicious": 1, "tech_tags": ["java"], "sentiment": "negative", "source_info": {"author_name": "test_author2", "repo_id": "test_repo2"}, "commit_type": "fix"}
        ] * 100 # Khoảng 200 mẫu giả
    except json.JSONDecodeError:
        raise ValueError(f"Lỗi giải mã JSON trong file: {json_path}")


    print(f"Loaded {len(samples)} samples from {json_path}")
    
    authors = set()
    repos = set()
    # Đảm bảo 'Unknown' tồn tại trong set để có index 0 nếu có sample không có author/repo
    authors.add('Unknown') 
    repos.add('Unknown')

    for item in samples:
        source_info = item.get('source_info', {})
        authors.add(source_info.get('author_name', 'Unknown'))
        repos.add(source_info.get('repo_id', 'Unknown'))
    
    author_map = {author: idx for idx, author in enumerate(sorted(list(authors)))}
    repo_map = {repo: idx for idx, repo in enumerate(sorted(list(repos)))}
    
    print(f"Found {len(author_map)} unique authors and {len(repo_map)} unique repositories")
    return samples, {'author_map': author_map, 'repo_map': repo_map}


def main():
    # Di chuyển khởi tạo device vào đây để tương thích với Windows khi num_workers > 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. Load training data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Sử dụng đường dẫn tương đối để tìm file dữ liệu
    # Giả định 'collected_data' nằm cùng cấp với thư mục chứa script này (ví dụ 'ai')
    # hoặc một cấp trên thư mục chứa script này.
    json_path_option1 = os.path.join(base_dir, "collected_data", "commit_messages_raw.json")
    json_path_option2 = os.path.join(os.path.dirname(base_dir), "collected_data", "commit_messages_raw.json")
    
    if os.path.exists(json_path_option1):
        json_path = json_path_option1
    elif os.path.exists(json_path_option2):
        json_path = json_path_option2
    else:
        # Nếu không tìm thấy, thử tạo đường dẫn mặc định như trong code gốc của bạn
        json_path = os.path.join(base_dir, "../collected_data/commit_messages_raw.json") # Thử đường dẫn này
        if not os.path.exists(json_path):
             # Nếu vẫn không thấy, sử dụng một đường dẫn cố định và chấp nhận có thể lỗi hoặc dùng dữ liệu giả
            print(f"Cảnh báo: Không tìm thấy commit_messages_raw.json ở các vị trí phổ biến. Sẽ thử đường dẫn gốc và có thể dùng dữ liệu giả nếu lỗi.")
            json_path = "backend/ai/collected_data/commit_messages_raw.json" # Đường dẫn gốc bạn dùng
            # raise FileNotFoundError(f"Không tìm thấy file dữ liệu huấn luyện ở: {json_path_option1} hoặc {json_path_option2} hoặc {json_path}")

    print("Loading data...")
    samples, mappings = load_commit_data(json_path)
    
    print("Shuffling dataset...")
    random.shuffle(samples) # Xáo trộn dữ liệu

    processor = TextProcessor()
    embed_loader = EmbeddingLoader(embedding_type='codebert') # Đảm bảo bạn đã cài đặt transformers và có model codebert
    try:
        embed_loader.load()
    except Exception as e:
        print(f"Lỗi khi tải embedding model: {e}. Kiểm tra lại cài đặt transformers và model name.")
        # Có thể thoát hoặc sử dụng embedding giả ở đây
        return


    # Split data
    print("Splitting data...")
    if not samples:
        print("Không có dữ liệu để chia. Kết thúc.")
        return
        
    train_size = int(0.8 * len(samples))
    if train_size < 1 and len(samples) > 0 : # Đảm bảo train_size ít nhất là 1 nếu có samples
        train_size = 1
    elif not samples: # Nếu không có samples nào cả
        print("Không có dữ liệu mẫu, không thể huấn luyện.")
        return

    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    # Create datasets
    # Kiểm tra train_samples và val_samples không rỗng trước khi tạo Dataset
    if not train_samples:
        print("Tập train rỗng. Không thể tạo DataLoader. Kiểm tra lại dữ liệu đầu vào.")
        return
        
    train_dataset = CommitDataset(
        train_samples, 
        processor, 
        embed_loader,
        author_map=mappings['author_map'],
        repo_map=mappings['repo_map']
    )
      # Optimize batch size based on GPU memory
    batch_size = 64 if device.type == 'cuda' else 8
    num_workers = 6 if device.type == 'cuda' else 0 # Giữ nguyên logic này
    
    # Chỉ tạo val_dataset và val_loader nếu có val_samples
    val_loader = None
    if val_samples:
        val_dataset = CommitDataset(
            val_samples,
            processor,
            embed_loader,
            author_map=mappings['author_map'],
            repo_map=mappings['repo_map']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    else:
        print("Tập validation rỗng. Sẽ không thực hiện validation.")


    # Optimize batch size based on GPU memory
    batch_size = 64 if device.type == 'cuda' else 8
    num_workers = 6 if device.type == 'cuda' else 0 # Giữ nguyên logic này

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    # val_loader đã được tạo ở trên nếu val_samples tồn tại

    # Initialize model
    print("Initializing model...")
    num_classes_dict = {
        'purpose': len(train_dataset.purpose_map), # Sử dụng kích thước map thay vì số cứng
        'tech_tag': len(train_dataset.tech_vocab),
        'suspicious': 2, # 0 hoặc 1
        'sentiment': len(train_dataset.sentiment_map),
        'author': len(mappings['author_map']) if mappings['author_map'] else 1, # Tránh lỗi nếu map rỗng
        'source_repo': len(mappings['repo_map']) if mappings['repo_map'] else 1,
        'commit_type': len(train_dataset.commit_type_map)
    }
      # Lấy kích thước embedding từ model config
    if embed_loader.embedding_type == 'codebert':
        embed_dim = embed_loader.model.config.hidden_size
    else:
        embed_dim = 768  # Giá trị mặc định
        
    model = HierarchicalAttentionNetwork(
        embed_dim=embed_dim,
        hidden_dim=128,
        num_classes_dict=num_classes_dict
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if device.type == 'cuda':
        print("Enabling CUDA optimizations...")
        torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)
    
    loss_fns = {
        'purpose': torch.nn.CrossEntropyLoss(),
        'suspicious': torch.nn.CrossEntropyLoss(),
        'tech_tag': torch.nn.CrossEntropyLoss(),
        'sentiment': torch.nn.CrossEntropyLoss(),
        'author': torch.nn.CrossEntropyLoss(),
        'source_repo': torch.nn.CrossEntropyLoss(),
        'commit_type': torch.nn.CrossEntropyLoss()
    }
    
    trainer = MultiTaskTrainer(model, optimizer, loss_fns, device=device)

    log_dir = os.path.join(base_dir, "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Training Configuration\n{'='*20}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of workers: {num_workers}\n")
        f.write(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_samples else 0}\n")
        f.write(f"Model classes: {num_classes_dict}\n\n")

    num_epochs = 10
    best_val_loss = float('inf')
    current_val_loss = float('inf') # Đổi tên biến để tránh nhầm lẫn với biến toàn cục nếu có
    
    print(f"Starting training on {device}...")
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = trainer.train_epoch(train_loader)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

            task_metrics = {}
            if val_loader: # Chỉ thực hiện validation nếu val_loader tồn tại
                current_val_loss, preds, true_labels = trainer.validate(val_loader)
                print(f"Epoch {epoch+1}: Val Loss = {current_val_loss:.4f}")

                for task in preds:
                    if len(np.unique(true_labels[task].numpy())) > 1:
                        metrics = calc_metrics(true_labels[task].numpy(), preds[task].argmax(-1).numpy())
                    else:
                        # Gán giá trị mặc định nếu chỉ có một lớp trong batch
                        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0} 
                    task_metrics[task] = metrics
            else: # Nếu không có val_loader, gán giá trị mặc định
                current_val_loss = float('inf') # Hoặc giá trị khác để không lưu model nếu không có val
                task_metrics = {task: {"f1": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0} for task in loss_fns.keys()}


            save_training_log(epoch + 1, train_loss, current_val_loss, task_metrics, log_file)

            if val_loader and current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss, # Sửa thành best_val_loss
                    'author_map': mappings['author_map'],
                    'repo_map': mappings['repo_map'],
                    'num_classes_dict': num_classes_dict # Lưu lại num_classes_dict
                }, os.path.join(base_dir, "models", "han_multitask_best.pth"))
                print(f"Epoch {epoch+1}: New best model saved with val_loss: {best_val_loss:.4f}")

            if device.type == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        import traceback
        traceback.print_exc() # In đầy đủ traceback để debug
    finally:
        print("Saving final model...")
        torch.save({
            'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'author_map': mappings['author_map'],
            'repo_map': mappings['repo_map'],
            'final_val_loss': current_val_loss,
            'num_classes_dict': num_classes_dict # Lưu lại num_classes_dict
        }, os.path.join(base_dir, "models", "han_multitask_final.pth"))

        print("Training finished!")
        if best_val_loss == float('inf') and val_loader:
             print("No best model was saved as validation loss did not improve or validation was not performed.")
        elif val_loader:
             print(f"Best validation loss: {best_val_loss:.4f}")
        else:
            print("Training completed without validation.")


if __name__ == "__main__":
    main()