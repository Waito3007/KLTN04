import json
import sys
from collections import Counter, defaultdict

MIN_TEXT_LEN = 5
REQUIRED_LABELS = ["task_type", "complexity", "technical_area", "required_skills", "priority"]

def assess_risk(files_changed, total_changes):
    if files_changed is None or total_changes is None:
        return "unknown"
    if files_changed <= 2 and total_changes <= 10:
        return "low"
    elif files_changed <= 5 and total_changes <= 50:
        return "medium"
    else:
        return "high"

def optimize_multimodal_sample(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        samples = obj["data"]
        meta = obj.get("metadata", {})
    elif isinstance(obj, list):
        samples = obj
        meta = {}
    else:
        print("File format not supported.")
        sys.exit(1)

    print(f"Tổng số sample gốc: {len(samples)}")
    filtered = []
    label_counter = defaultdict(Counter)
    risk_counter = Counter()
    for s in samples:
        text = s.get("text", "")
        labels = s.get("labels", {})
        # Loại sample thiếu nhãn chính hoặc text quá ngắn
        if len(text) < MIN_TEXT_LEN or not all(l in labels and labels[l] for l in REQUIRED_LABELS):
            continue
        # Xóa các trường không cần thiết trong metadata
        if "metadata" in s:
            for k in ["commit_id", "author_email", "timestamp"]:
                if k in s["metadata"]:
                    del s["metadata"][k]
        # Đảm bảo features tồn tại
        if "features" not in s:
            s["features"] = {}
        # Xóa các trường không cần thiết trong features
        REMOVE_FEATURES = ["commit_id", "author", "author_email", "timestamp", "repository"]
        for k in REMOVE_FEATURES:
            if k in s["features"]:
                del s["features"][k]
        # Đánh giá nguy cơ
        meta_s = s.get("metadata", {})
        files_changed = meta_s.get("files_changed")
        total_changes = meta_s.get("total_changes")
        s["features"]["risk_level"] = assess_risk(files_changed, total_changes)
        # Thống kê nhãn
        for k, v in labels.items():
            if isinstance(v, list):
                for item in v:
                    label_counter[k][item] += 1
            else:
                label_counter[k][v] += 1
        risk_counter[s["features"]["risk_level"]] += 1
        filtered.append(s)

    print(f"Số sample sau lọc: {len(filtered)}")
    print("Phân bố risk_level:", dict(risk_counter))
    for k in REQUIRED_LABELS:
        print(f"Phân bố nhãn {k}:", dict(label_counter[k]))

    # Lưu file tối ưu
    if isinstance(obj, dict) and "data" in obj:
        obj["data"] = filtered
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu file tối ưu: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python optimize_multimodal_sample.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    optimize_multimodal_sample(input_file, output_file)
