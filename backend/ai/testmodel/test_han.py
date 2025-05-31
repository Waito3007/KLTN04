import torch
import sys
# Add workspace root to Python path
sys.path.append("c:\\SAN\\KLTN\\KLTN04\\backend")
sys.path.append("c:\\SAN\\KLTN\\KLTN04\\backend\\ai")
from ai.models.hierarchical_attention import HierarchicalAttentionNetwork
from data_preprocessing.text_processor import TextProcessor

# Load the trained HAN model
model_path = "c:\\SAN\\KLTN\\KLTN04\\backend\\ai\\models\\han_multitask_best.pth"
model = HierarchicalAttentionNetwork(embed_dim=768, hidden_dim=128, num_classes_dict={
    'purpose': 10, 'suspicious': 2, 'tech_tag': 15, 'sentiment': 3, 'author': 20, 'source_repo': 5
})

# Extract model_state_dict from checkpoint
checkpoint = torch.load(model_path)

# Extract num_classes_dict from checkpoint
num_classes_dict = checkpoint.get('num_classes_dict', {
    'purpose': 9, 'suspicious': 2, 'tech_tag': 79, 'sentiment': 3, 'author': 7, 'source_repo': 7
})

# Reinitialize model with correct num_classes_dict
model = HierarchicalAttentionNetwork(embed_dim=768, hidden_dim=128, num_classes_dict=num_classes_dict)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize TextProcessor
text_processor = TextProcessor()

# Define a word2idx mapping for token conversion
word2idx = {'<unk>': 0}  # Add actual word-to-index mapping here

# Sample commit messages for testing
commit_messages = [
    "feat: Thêm tính năng xử lý lỗi",
    "fix: Sửa lỗi hiển thị trên giao diện người dùng",
    "docs: Cập nhật tài liệu API",
    "refactor: Tối ưu code module tìm kiếm",
    "style: Chỉnh sửa format code",
    "test: Thêm test case cho hàm login",
    "chore: Cập nhật dependencies",
    "uncategorized: Thay đổi cấu trúc thư mục",
    "feat: Thêm API thanh toán mới",
    "fix: Sửa layout bị cách ra ngoài",
]

# Load or generate embeddings for words
embedding_dim = 768  # Define embedding dimension
embedding_loader = torch.nn.Embedding(len(word2idx), embedding_dim)

# Ensure input tensor has the correct shape
max_sent_len = 10  # Define maximum number of sentences
max_word_len = 20  # Define maximum number of words per sentence

# Define label mappings for each task
label_mappings = {
    'purpose': ['Thêm tính năng', 'Sửa lỗi', 'Cập nhật tài liệu', 'Tối ưu code', 'Chỉnh sửa format', 'Thêm test case', 'Cập nhật dependencies', 'Thay đổi cấu trúc', 'Thêm API mới', 'Khác'],
    'sentiment': ['Tích cực', 'Tiêu cực', 'Trung lập'],
    'tech_tag': ['Python', 'JavaScript', 'HTML', 'CSS', 'SQL', 'Docker', 'React', 'Angular', 'Vue', 'Khác'],
    # Add mappings for other tasks as needed
}

# Preprocess and predict
for message in commit_messages:
    processed_message = text_processor.process_document(message, word2idx=word2idx)
    # Pad processed_message to match expected dimensions
    processed_message = [sent + [0]*(max_word_len-len(sent)) if len(sent)<max_word_len else sent for sent in processed_message]
    if len(processed_message) < max_sent_len:
        processed_message += [[0]*max_word_len]*(max_sent_len-len(processed_message))

    # Convert processed_message to embeddings
    embedded_message = [embedding_loader(torch.tensor(sent)) for sent in processed_message]
    embedded_message = torch.stack(embedded_message)

    # Ensure input tensor has the correct shape
    input_tensor = embedded_message.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(input_tensor)

    # Extract predictions dictionary from model output
    predictions_dict = predictions[0]  # First element contains the task-specific outputs

    # Convert predictions to human-readable labels
    readable_predictions = {}
    for task, output in predictions_dict.items():
        predicted_label_idx = torch.argmax(output).item()
        # Ensure the index is within the range of label mappings
        if predicted_label_idx < len(label_mappings.get(task, [])):
            readable_predictions[task] = label_mappings.get(task, ['Unknown'])[predicted_label_idx]
        else:
            readable_predictions[task] = 'Unknown'

    print(f"Commit message: {message}")
    print("Predictions:")
    for task, label in readable_predictions.items():
        print(f"  {task}: {label}")
    print()
