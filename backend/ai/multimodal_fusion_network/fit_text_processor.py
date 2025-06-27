import json
from data_processing.text_processor import TextProcessor

# Đường dẫn tới file train
train_path = "data/processed/train.json"
processor_path = "data/processed/text_processor.json"

# Đọc dữ liệu train
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
texts = [sample["text"] for sample in train_data["data"] if "text" in sample]

# Fit TextProcessor
text_processor = TextProcessor(max_vocab_size=10000, max_sequence_length=100)
text_processor.fit(texts)
text_processor.save(processor_path)
print(f"Đã fit và lưu TextProcessor vào {processor_path}")