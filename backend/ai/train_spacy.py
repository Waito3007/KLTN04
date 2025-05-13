import spacy
from spacy.training.example import Example
import random
import json

# Tải mô hình đa ngôn ngữ
nlp = spacy.load("xx_ent_wiki_sm")

# Thêm textcat vào pipeline nếu chưa có
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)
else:
    textcat = nlp.get_pipe("textcat")

# Thêm các nhãn vào mô hình
textcat.add_label("feat")
textcat.add_label("fix")
textcat.add_label("docs")
textcat.add_label("style")
textcat.add_label("refactor")
textcat.add_label("chore")
textcat.add_label("test")
textcat.add_label("uncategorized")

# Đọc dữ liệu từ file JSON
with open('train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Chuyển dữ liệu thành định dạng spaCy
train_data = []

for entry in data:
    text = entry["text"]
    cats = entry["cats"]
    
    # Tạo Example từ văn bản và nhãn
    example = Example.from_dict(nlp.make_doc(text), {"cats": cats})
    train_data.append(example)

# Huấn luyện mô hình
optimizer = nlp.begin_training()
for epoch in range(10):  # Số epoch có thể điều chỉnh
    random.shuffle(train_data)  # Shuffle dữ liệu mỗi epoch
    losses = {}
    for example in train_data:
        # Cập nhật mô hình
        nlp.update([example], losses=losses)
    print(f"Epoch {epoch} - Losses: {losses}")

# Lưu mô hình đã huấn luyện
output_dir = "modelAi"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")
