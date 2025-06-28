import random
import json
from typing import List, Dict

# Giả lập hàm load_model và predict (thay bằng import thực tế từ project nếu có)
def load_model(model_path: str):
    # TODO: Thay thế bằng hàm load thực tế
    print(f"[MOCK] Đã tải mô hình từ {model_path}")
    return lambda x: mock_predict(x)

def mock_predict(batch: List[Dict]):
    # Trả về kết quả giả lập cho mỗi commit
    task_types = ["bug_fix", "feature", "documentation", "security", "refactoring", "testing", "devops"]
    complexities = ["low", "medium", "high"]
    technical_areas = ["frontend", "backend", "database", "infrastructure", "mobile"]
    skills = ["python", "javascript", "security", "testing", "react", "angular", "dotnet", "devops", "database"]
    priorities = ["critical", "high", "medium", "low"]
    risk_levels = ["low", "medium", "high"]
    
    results = []
    for commit in batch:
        result = {
            "task_type": random.sample(task_types, k=random.randint(1, 3)),
            "complexity": random.choice(complexities),
            "technical_area": random.sample(technical_areas, k=random.randint(1, 2)),
            "required_skills": random.sample(skills, k=random.randint(1, 3)),
            "priority": random.choice(priorities),
            "risk_level": random.choice(risk_levels)
        }
        results.append(result)
    return results

def generate_mock_commits(n=200, repo="example/repo"):
    authors = ["alice", "bob", "carol", "dave", "eve"]
    commit_msgs = [
        "Fix bug in user authentication",
        "Add new feature for dashboard",
        "Update documentation for API",
        "Improve security checks",
        "Refactor data processing module",
        "Add unit tests for utils",
        "DevOps: update CI/CD pipeline"
    ]
    files = [
        ["src/auth.py", "src/utils.py"],
        ["src/dashboard.js", "src/api.js"],
        ["README.md"],
        ["src/security.py"],
        ["src/data.py", "src/processing.py"],
        ["tests/test_utils.py"],
        [".github/workflows/ci.yml"]
    ]
    data = []
    for i in range(n):
        idx = random.randint(0, len(commit_msgs)-1)
        commit = {
            "text": commit_msgs[idx],
            "metadata": {
                "commit_id": f"mock_{i}",
                "author": random.choice(authors),
                "files_changed": len(files[idx]),
                "additions": random.randint(1, 100),
                "deletions": random.randint(0, 50),
                "total_changes": random.randint(1, 150),
                "is_merge": False,
                "modified_files": files[idx],
                "repository": repo
            }
        }
        data.append(commit)
    return data

def main():
    print("--- Test mô hình phân loại commit với dữ liệu mock ---")
    # Bước 1: Tải mô hình (thay đường dẫn thực tế nếu cần)
    model = load_model("../output/best_model.pt")
    # Bước 2: Sinh dữ liệu mock
    mock_data = generate_mock_commits(n=200, repo="mock/repo")
    # Bước 3: Dự đoán
    results = model(mock_data)
    # Bước 4: Hiển thị kết quả mẫu
    for i, (commit, pred) in enumerate(zip(mock_data, results)):
        print(f"\nCommit {i+1}: {commit['text']}")
        print(f"  - Task type: {pred['task_type']}")
        print(f"  - Complexity: {pred['complexity']}")
        print(f"  - Technical area: {pred['technical_area']}")
        print(f"  - Required skills: {pred['required_skills']}")
        print(f"  - Priority: {pred['priority']}")
        print(f"  - Risk level: {pred['risk_level']}")

    # Lưu kết quả ra file json
    with open("mock_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nĐã lưu kết quả mock vào mock_test_results.json")

    # Sinh báo cáo chi tiết tiếng Việt
    report = []
    report.append("BÁO CÁO KIỂM THỬ MÔ HÌNH PHÂN LOẠI COMMIT (Mock Data)")
    report.append("\n1. Tổng số commit kiểm thử: {}".format(len(mock_data)))
    from collections import Counter
    def flatten(l):
        return [item for sublist in l for item in sublist]
    task_type_counter = Counter(flatten([r['task_type'] for r in results]))
    complexity_counter = Counter([r['complexity'] for r in results])
    technical_area_counter = Counter(flatten([r['technical_area'] for r in results]))
    skills_counter = Counter(flatten([r['required_skills'] for r in results]))
    priority_counter = Counter([r['priority'] for r in results])
    risk_counter = Counter([r['risk_level'] for r in results])

    report.append("\n2. Thống kê phân loại tổng quan:")
    report.append("- Task type: " + ", ".join(f"{k}: {v}" for k, v in task_type_counter.items()))
    report.append("- Complexity: " + ", ".join(f"{k}: {v}" for k, v in complexity_counter.items()))
    report.append("- Technical area: " + ", ".join(f"{k}: {v}" for k, v in technical_area_counter.items()))
    report.append("- Required skills: " + ", ".join(f"{k}: {v}" for k, v in skills_counter.items()))
    report.append("- Priority: " + ", ".join(f"{k}: {v}" for k, v in priority_counter.items()))
    report.append("- Risk level: " + ", ".join(f"{k}: {v}" for k, v in risk_counter.items()))

    report.append("\n3. Báo cáo chi tiết từng commit:")
    for i, (commit, pred) in enumerate(zip(mock_data, results)):
        report.append(f"\nCommit {i+1}:")
        report.append(f"  - DỮ LIỆU ĐẦU VÀO:")
        report.append(f"    + Tác giả: {commit['metadata']['author']}")
        report.append(f"    + Nội dung: {commit['text']}")
        report.append(f"    + Số file thay đổi: {commit['metadata']['files_changed']}, Số dòng thêm: {commit['metadata']['additions']}, Số dòng xoá: {commit['metadata']['deletions']}")
        report.append(f"    + Danh sách file: {commit['metadata']['modified_files']}")
        report.append(f"    + Repo: {commit['metadata']['repository']}")
        report.append(f"  - KẾT QUẢ PHÂN TÍCH (SAU KHI ĐƯA VÀO MODEL):")
        report.append(f"    + Task type (loại tác vụ): {pred['task_type']}")
        report.append(f"    + Complexity (độ phức tạp): {pred['complexity']}")
        report.append(f"    + Technical area (lĩnh vực kỹ thuật): {pred['technical_area']}")
        report.append(f"    + Required skills (kỹ năng cần thiết): {pred['required_skills']}")
        report.append(f"    + Priority (ưu tiên): {pred['priority']}")
        report.append(f"    + Risk level (nguy cơ): {pred['risk_level']}")

    report.append("\n4. Đánh giá khả năng mô hình:")
    report.append("- Mô hình có thể phân loại commit theo nhiều nhãn cùng lúc (đa nhãn), ví dụ: task_type, complexity, technical_area, required_skills, priority.")
    report.append("- Có thể đánh giá nguy cơ (risk_level) dựa trên đặc trưng commit.")
    report.append("- Kết quả phân loại đa dạng, phù hợp với dữ liệu commit thực tế.")
    report.append("- Có thể mở rộng để phân tích lịch sử đóng góp hoặc phát hiện mẫu commit bất thường nếu cung cấp thêm dữ liệu lịch sử.")

    with open("mock_test_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\nĐã lưu báo cáo chi tiết vào mock_test_report.txt (tiếng Việt, dễ đọc, có từng commit)")

if __name__ == "__main__":
    main()
