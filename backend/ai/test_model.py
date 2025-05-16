import spacy

# Load mô hình đã train
nlp = spacy.load("modelAi")

# Các câu commit cần test
texts = [
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

for text in texts:
    doc = nlp(text)
    print(f"Văn bản: {text}")
    for label, score in doc.cats.items():
        if score > 0.05:  # Tùy chỉnh ngưỡng lọc nhãn
            print(f"{label}: {score:.4f}")
    print()
