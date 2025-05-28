import os
import re
import glob
import json

def extract_code_comments_from_file(filepath):
    comments = []
    ext = os.path.splitext(filepath)[1]
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return comments  # Bỏ qua file không đọc được
    if ext in ['.py']:
        for line in lines:
            match = re.match(r'\s*#(.*)', line)
            if match:
                comments.append(match.group(1).strip())
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        in_block = False
        for line in lines:
            if '/*' in line:
                in_block = True
            if in_block:
                comments.append(line.strip())
            if '*/' in line:
                in_block = False
            match = re.match(r'\s*//(.*)', line)
            if match:
                comments.append(match.group(1).strip())
    return comments

def extract_comments_from_repo(repo_path, exts=['.py', '.js', '.jsx', '.ts', '.tsx']):
    all_comments = []
    for ext in exts:
        for filepath in glob.glob(os.path.join(repo_path, f'**/*{ext}'), recursive=True):
            comments = extract_code_comments_from_file(filepath)
            for c in comments:
                if c:
                    all_comments.append({'text': c, 'source': filepath})
    return all_comments

def save_comments_to_json(comments, out_path, repo_name=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = []
    for idx, c in enumerate(comments):
        output.append({
            "id": f"comment_{idx}",
            "data_type": "code_comment",
            "raw_text": c["text"] if isinstance(c, dict) else c,
            "source_info": {
                "repo_name": repo_name or "KLTN04",
                "file_path": c["source"] if isinstance(c, dict) and "source" in c else None
            },
            "labels": {
                "purpose": None,
                "suspicious": None,
                "tech_tag": None,
                "sentiment": None
            }
        })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    repo_path = '../../backend'  # Giới hạn quét trong backend
    comments = extract_comments_from_repo(repo_path)
    save_comments_to_json(comments, '../collected_data/code_comments.json')
    print(f'Extracted {len(comments)} comments.')
