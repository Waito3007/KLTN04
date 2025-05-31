import torch
import numpy as np
import sys
# Add workspace root to Python path
sys.path.append("c:\\SAN\\KLTN\\KLTN04\\backend") # Điều chỉnh nếu cần
sys.path.append("c:\\SAN\\KLTN\\KLTN04\\backend\\ai") # Điều chỉnh nếu cần
from ai.models.hierarchical_attention import HierarchicalAttentionNetwork
from ai.data_preprocessing.text_processor import TextProcessor
from ai.data_preprocessing.embedding_loader import EmbeddingLoader # Import thêm EmbeddingLoader

# Load the trained HAN model
# Đảm bảo đường dẫn này chính xác tới file model của bạn
model_path = "c:\\SAN\\KLTN\\KLTN04\\backend\\ai\\models\\han_multitask_best.pth" 

# --- PHẦN CẦN ĐIỀU CHỈNH ĐỂ TEST CHÍNH XÁC ---
# 1. Khởi tạo TextProcessor và EmbeddingLoader giống như trong training
processor = TextProcessor() # max_sent_len và max_word_len mặc định là 30, 20
embed_loader = EmbeddingLoader(embedding_type='codebert')
try:
    embed_loader.load()
    print("CodeBERT embedding loaded successfully for testing.")
except Exception as e:
    print(f"Error loading CodeBERT model: {e}")
    print("Please ensure CodeBERT model is accessible. Testing may not work correctly.")
    # Có thể thoát hoặc dùng embedding giả để test cấu trúc model
    # For now, we'll let it proceed, but embeddings might be zero if loader failed.

# 2. Tải checkpoint và num_classes_dict từ model đã lưu
try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu')) # Tải lên CPU trước
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file model tại: {model_path}")
    print("Vui lòng kiểm tra lại đường dẫn và đảm bảo bạn đã huấn luyện và lưu model.")
    sys.exit(1) # Thoát nếu không có model

# Lấy num_classes_dict từ checkpoint
# Nếu không có trong checkpoint, bạn cần đảm bảo nó khớp với lúc train
default_num_classes = {
    'purpose': 9, 'tech_tag': 79, 'suspicious': 2, 
    'sentiment': 3, 'author': 1, 'source_repo': 1, 'commit_type': 8
}
num_classes_dict_from_checkpoint = checkpoint.get('num_classes_dict', default_num_classes)
print(f"Number of classes from checkpoint: {num_classes_dict_from_checkpoint}")

# Khởi tạo model với num_classes_dict chính xác từ checkpoint
model = HierarchicalAttentionNetwork(
    embed_dim=embed_loader.model.config.hidden_size if embed_loader.embedding_type == 'codebert' and hasattr(embed_loader.model, 'config') and hasattr(embed_loader.model.config, 'hidden_size') else 768, # Lấy embed_dim từ model config
    hidden_dim=128, 
    num_classes_dict=num_classes_dict_from_checkpoint
)

# Tải state_dict vào model
# Xử lý trường hợp model được lưu với DataParallel (có prefix 'module.')
model_state_dict = checkpoint['model_state_dict']
if list(model_state_dict.keys())[0].startswith('module.'):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] # Xóa `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(model_state_dict)

model.eval() # Chuyển model sang chế độ evaluation
print("Model loaded successfully for testing.")

# Định nghĩa các mapping nhãn để hiển thị kết quả (giữ nguyên hoặc điều chỉnh cho khớp)
# Đây là ví dụ, bạn cần đảm bảo chúng khớp với cách bạn định nghĩa nhãn khi training
# Hoặc tốt hơn, tải author_map và repo_map từ checkpoint nếu có
author_map_from_checkpoint = checkpoint.get('author_map', {'Unknown': 0})
repo_map_from_checkpoint = checkpoint.get('repo_map', {'Unknown': 0})

# Tạo ngược lại id -> label cho author và repo để dễ đọc kết quả
idx2author = {idx: author for author, idx in author_map_from_checkpoint.items()}
idx2repo = {idx: repo for repo, idx in repo_map_from_checkpoint.items()}

# Các mapping khác có thể lấy từ định nghĩa trong CommitDataset nếu cần
# Ví dụ:
# purpose_idx_to_label = {v: k for k, v in CommitDataset(...).purpose_map.items()}
# sentiment_idx_to_label = {v: k for k, v in CommitDataset(...).sentiment_map.items()}
# tech_vocab_list = CommitDataset(...).tech_vocab # list of tech tags
# commit_type_idx_to_label = {v: k for k, v in CommitDataset(...).commit_type_map.items()}

# Ví dụ đơn giản hóa cho mục đích hiển thị
label_mappings_display = {
    'purpose': ['Feature Impl.', 'Bug Fix', 'Refactor', 'Doc Update', 'Test Update', 'Security Patch', 'Code Style', 'Build/CI/CD', 'Other'],
    'sentiment': ['Positive', 'Neutral', 'Negative'],
    'suspicious': ['Not Suspicious', 'Suspicious'],
    'commit_type': ['feat', 'fix', 'docs', 'refactor', 'style', 'test', 'chore', 'uncategorized']
    # 'tech_tag': tech_vocab_list, # Sẽ là một list dài
    # 'author': idx2author, # Sẽ là một dict
    # 'source_repo': idx2repo # Sẽ là một dict
}


# Sample commit messages for testing
commit_messages = [
    "feat: Thêm tính năng xử lý lỗi và đăng nhập cho người dùng mới",
    "fix: Sửa lỗi hiển thị trên giao diện người dùng ở phần thanh toán",
    "docs: Cập nhật tài liệu API cho module products",
    "refactor: Tối ưu code module tìm kiếm và cải thiện tốc độ query",
    "style: Chỉnh sửa format code theo chuẩn PEP8 cho toàn bộ project python",
    "test: Thêm test case cho hàm login và đăng ký bằng pytest",
    "chore: Cập nhật dependencies và cấu hình CI/CD cho Docker",
    "uncategorized: Thay đổi cấu trúc thư mục và một vài điều chỉnh nhỏ",
    "feat: Thêm API thanh toán mới sử dụng Stripe và vue",
    "fix(UI): Sửa layout bị cách ra ngoài màn hình dashboard react"
]

# Preprocess and predict
print("\n--- Starting Predictions ---")
for message in commit_messages:
    # 3. Xử lý văn bản và tạo embedding giống như trong CommitDataset
    doc_tokens = processor.process_document(message) # Không dùng word2idx
    
    embed_dim_model = embed_loader.model.config.hidden_size if embed_loader.embedding_type == 'codebert' and hasattr(embed_loader.model, 'config') and hasattr(embed_loader.model.config, 'hidden_size') else 768
    embed_doc = np.zeros((processor.max_sent_len, processor.max_word_len, embed_dim_model), dtype=np.float32)

    for i, sent_tokens in enumerate(doc_tokens):
        if i >= processor.max_sent_len: break
        for j, word_token in enumerate(sent_tokens):
            if j >= processor.max_word_len: break
            word_str = str(word_token)
            # Lấy embedding từ embed_loader đã load CodeBERT
            embed_doc[i, j] = embed_loader.get_word_embedding(word_str) 
                                           # (Cần đảm bảo get_word_embedding xử lý từ không có trong vocab của CodeBERT,
                                           # ví dụ trả về vector zero hoặc vector của token UNK)
    
    input_tensor = torch.tensor(embed_doc, dtype=torch.float32).unsqueeze(0) # Thêm batch dimension

    # Chuyển input_tensor tới device của model nếu cần (model đã ở CPU sau map_location)
    # model.to(device) # Nếu bạn muốn chạy model trên GPU, hãy chuyển model và input sang GPU
    # input_tensor = input_tensor.to(device)

    with torch.no_grad():
        predictions_output, word_attn, sent_attn = model(input_tensor) # Model trả về tuple

    print(f"\nCommit message: {message}")
    print("Predictions:")
    for task, output_tensor in predictions_output.items():
        predicted_idx = torch.argmax(output_tensor, dim=1).item()
        
        # Hiển thị nhãn dựa trên mapping
        if task in label_mappings_display:
            if isinstance(label_mappings_display[task], list): # Cho purpose, sentiment, suspicious, commit_type
                 if 0 <= predicted_idx < len(label_mappings_display[task]):
                    readable_label = label_mappings_display[task][predicted_idx]
                 else:
                    readable_label = f"Unknown_Index ({predicted_idx})"
            # elif isinstance(label_mappings_display[task], dict): # Cho author, source_repo (nếu bạn map ngược lại)
            #     readable_label = label_mappings_display[task].get(predicted_idx, f"Unknown_ID ({predicted_idx})")
            # else: # Cho tech_tag (list dài)
            #      if 0 <= predicted_idx < len(label_mappings_display[task]):
            #         readable_label = label_mappings_display[task][predicted_idx]
            #      else:
            #         readable_label = f"Unknown_Tech_Index ({predicted_idx})"

        # Đơn giản hóa cho các task còn lại chưa có mapping hiển thị chi tiết
        elif task == 'tech_tag':
            # Giả sử tech_vocab_list đã được lấy từ checkpoint hoặc CommitDataset
            tech_vocab_list_example = ['python', 'javascript', 'java', 'docker', 'react'] # Thay bằng list thật
            if 0 <= predicted_idx < len(tech_vocab_list_example):
                 readable_label = tech_vocab_list_example[predicted_idx]
            else:
                readable_label = f"Tech_Index_{predicted_idx}"
        elif task == 'author':
            readable_label = idx2author.get(predicted_idx, f"Author_ID_{predicted_idx}")
        elif task == 'source_repo':
            readable_label = idx2repo.get(predicted_idx, f"Repo_ID_{predicted_idx}")
        else:
            readable_label = f"Index_{predicted_idx}"
            
        print(f"  {task}: {readable_label} (Raw output: {output_tensor.softmax(dim=1).max().item():.4f})")