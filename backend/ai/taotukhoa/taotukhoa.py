import json
from collections import defaultdict

# Danh sách nhãn
categories = [
    "auth", "search", "cart", "order", "profile", "product",
    "api", "ui", "notification", "dashboard", "fix", "feat",
    "refactor", "test", "style", "docs", "chore", "uncategorized"
]

# Từ khóa mẫu theo từng category
keyword_seed = {
    "auth": ["login", "logout", "register", "authentication", "authorization", "signin", "signup", "verify", "jwt", "token", "mã xác thực", "đăng nhập", "đăng ký"],
    "search": ["search", "filter", "query", "lookup", "tìm kiếm", "lọc"],
    "cart": ["cart", "add to cart", "remove from cart", "giỏ hàng", "thêm vào giỏ", "xóa khỏi giỏ"],
    "order": ["order", "checkout", "invoice", "bill", "đặt hàng", "thanh toán", "hóa đơn", "giao hàng"],
    "profile": ["profile", "account", "user info", "thông tin cá nhân", "hồ sơ", "avatar", "user"],
    "product": ["product", "item", "goods", "sản phẩm", "thêm sản phẩm", "chỉnh sửa sản phẩm"],
    "api": ["api", "endpoint", "request", "response", "swagger", "rest", "graphql"],
    "ui": ["ui", "interface", "layout", "giao diện", "font", "màu sắc", "hiển thị", "style", "responsive", "theme"],
    "notification": ["notification", "alert", "thông báo", "popup", "toast"],
    "dashboard": ["dashboard", "admin", "panel", "thống kê", "biểu đồ", "quản trị", "báo cáo"],

    "fix": ["fix", "bug", "sửa lỗi", "resolve", "patch", "hotfix", "lỗi", "error", "issue"],
    "feat": ["feat", "feature", "thêm chức năng", "implement", "function", "module", "new feature"],
    "refactor": ["refactor", "optimize", "tối ưu", "clean code", "cải tiến", "tái cấu trúc", "improve"],
    "test": ["test", "unit test", "integration test", "test case", "kiểm thử", "viết test"],
    "style": ["style", "format", "prettier", "indent", "reformat", "format lại", "chỉnh code"],
    "docs": ["docs", "readme", "tài liệu", "viết tài liệu", "hướng dẫn", "documentation"],
    "chore": ["chore", "cấu hình", "setup", "config", "update dependencies", "maintenance", "tooling"],
    "uncategorized": ["misc", "other", "khác", "thay đổi cấu trúc", "cập nhật"]
}

# Mở rộng mỗi nhãn thành nhiều từ khóa (dựa trên các gốc từ)
def expand_keywords(seed):
    expanded = set()
    for word in seed:
        expanded.add(word.lower())
        if " " in word:
            expanded.add(word.replace(" ", "-"))
            expanded.add(word.replace(" ", "_"))
        if word.isalpha():
            expanded.add(word + "s")
            expanded.add(word + "ed")
            expanded.add(word + "ing")
    return list(expanded)

# Xây dựng dict tổng hợp
keyword_dict = defaultdict(list)

for category, seed_words in keyword_seed.items():
    keyword_dict[category] = expand_keywords(seed_words)

# Xuất ra file JSON
with open("subintent_keywords.json", "w", encoding="utf-8") as f:
    json.dump(keyword_dict, f, ensure_ascii=False, indent=2)

print("✅ Đã tạo file 'subintent_keywords.json' với từ khóa mở rộng.")
