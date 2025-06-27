import json
from data_processing.metadata_processor import MetadataProcessor

# Đường dẫn tới file train
train_path = "data/processed/train.json"
meta_processor_path = "data/processed/metadata_processor.json"

# Đọc dữ liệu train
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)
features = [sample["features"] for sample in train_data["data"] if "features" in sample]

# Fit MetadataProcessor
metadata_processor = MetadataProcessor()
metadata_processor.fit(features)
metadata_processor.save(meta_processor_path)
print(f"Đã fit và lưu MetadataProcessor vào {meta_processor_path}")