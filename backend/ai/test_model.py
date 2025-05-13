import spacy

# Tải mô hình đã huấn luyện
nlp = spacy.load("modelAi")

# Văn bản cần phân loại
texts = [
    "feat: Thêm tính năng xử lý lỗi",
    "fix: Sửa lỗi hiển thị trên giao diện người dùng",
    "docs: Cập nhật tài liệu API",
    "feat: Thêm tính năng tìm kiếm",
    "refactor: Cải tiến tính năng tìm kiếm"
]

# Dùng mô hình để phân loại văn bản
for text in texts:
    doc = nlp(text)
    print(f"Văn bản: {text}")
    for label, score in doc.cats.items():
        print(f"{label}: {score}")
    print()
