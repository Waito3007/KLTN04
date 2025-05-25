import spacy
from spacy.training.example import Example
import random
import json

# Tải mô hình đa ngôn ngữ
nlp = spacy.load("xx_ent_wiki_sm")

# Thêm textcat_multilabel vào pipeline nếu chưa có
if "textcat_multilabel" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat_multilabel", last=True)
else:
    textcat = nlp.get_pipe("textcat_multilabel")

# Thêm các nhãn
labels = ["feat", "fix", "docs", "style", "refactor", "chore", "test", "uncategorized",
          "auth", "search", "cart", "order", "profile", "product", "api", "ui", "notification", "dashboard"]

for label in labels:
    textcat.add_label(label)

# Đọc dữ liệu từ file JSON
with open('train_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train_data = []
for entry in data:
    text = entry["text"]
    cats = entry["cats"]
    example = Example.from_dict(nlp.make_doc(text), {"cats": cats})
    train_data.append(example)

optimizer = nlp.begin_training()
for epoch in range(10):
    random.shuffle(train_data)
    losses = {}
    for example in train_data:
        nlp.update([example], losses=losses)
    print(f"Epoch {epoch} - Losses: {losses}")

output_dir = "modelAi"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")
