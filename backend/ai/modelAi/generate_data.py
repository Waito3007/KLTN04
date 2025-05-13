import json
from db.database import database
from db.models.commits import commits  # Đảm bảo bạn đã import đúng model commits
from datetime import datetime

# Hàm để chuyển dữ liệu commit thành định dạng spaCy
async def convert_to_spacy_format():
    # Kết nối cơ sở dữ liệu trước khi truy vấn
    await database.connect()

    # Truy vấn tất cả dữ liệu commits từ database
    query = commits.select()  # Thay đổi nếu bạn sử dụng ORM khác như SQLAlchemy
    rows = await database.fetch_all(query)

    # Danh sách để lưu dữ liệu theo định dạng spaCy
    data = []

    for row in rows:
        # Lấy dữ liệu từ commit
        sha = row["sha"]
        message = row["message"]  # Lấy trường 'message' từ commit
        author_name = row["author_name"]
        author_email = row["author_email"]
        date = row["date"]  # Trường ngày giờ commit
        insertions = row["insertions"]
        deletions = row["deletions"]
        files_changed = row["files_changed"]
        repo_id = row["repo_id"]

        # Nếu date là kiểu datetime, chuyển nó thành chuỗi
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d %H:%M:%S')  # Chuyển đổi theo định dạng mong muốn
        
        # Kiểm tra commit có chứa trường message hay không
        if message:
            # Phân loại label từ message
            label = get_label_from_message(message)

            # Thêm vào danh sách data với định dạng spaCy
            data.append({
                "text": message,  # Nội dung commit
                "meta": {
                    "sha": sha,  # SHA của commit
                    "author_name": author_name,  # Tên tác giả
                    "author_email": author_email,  # Email tác giả
                    "date": date,  # Ngày commit (chuỗi)
                    "insertions": insertions,  # Số dòng thêm
                    "deletions": deletions,  # Số dòng xóa
                    "files_changed": files_changed,  # Số file thay đổi
                    "repo_id": repo_id  # ID của repository
                },
                "cats": label  # Thẻ phân loại của commit
            })

    # Ghi dữ liệu ra file JSON cho spaCy
    with open("ai/train_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Dữ liệu đã được chuyển thành công!")

    # Ngắt kết nối cơ sở dữ liệu sau khi hoàn thành
    await database.disconnect()

# Hàm phân loại label từ message (cần phải viết logic phân loại của bạn)
def get_label_from_message(message):
    # Các từ khóa phổ biến trong commit
    categories = {
        "feat": {"feat": 1.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "fix": {"feat": 0.0, "fix": 1.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "docs": {"feat": 0.0, "fix": 0.0, "docs": 1.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "style": {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 1.0, "refactor": 0.0, "chore": 0.0, "test": 0.0},
        "refactor": {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 1.0, "chore": 0.0, "test": 0.0},
        "chore": {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 1.0, "test": 0.0},
        "test": {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 1.0},
    }

    # Kiểm tra nếu commit chứa một trong các từ khóa trên
    for keyword, label in categories.items():
        if keyword in message.lower():
            return label
    
    # Nếu không có từ khóa nào trong message, phân loại là 'uncategorized'
    return {"feat": 0.0, "fix": 0.0, "docs": 0.0, "style": 0.0, "refactor": 0.0, "chore": 0.0, "test": 0.0, "uncategorized": 1.0}

# Chạy hàm convert_to_spacy_format
if __name__ == "__main__":
    import asyncio
    asyncio.run(convert_to_spacy_format())
