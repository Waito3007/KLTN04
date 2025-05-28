import os
import json

def extract_readme(repo_path):
    readme_path = None
    for fname in os.listdir(repo_path):
        if fname.lower().startswith('readme'):
            readme_path = os.path.join(repo_path, fname)
            break
    if not readme_path or not os.path.isfile(readme_path):
        return None
    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return content

def save_readme_to_json(content, out_path, repo_name=None):
    output = [{
        "id": "readme_1",
        "data_type": "readme_content",
        "raw_text": content,
        "source_info": {
            "repo_name": repo_name or "KLTN04",
            "file_path": "README.md"
        },
        "labels": {
            "purpose": None,
            "suspicious": None,
            "tech_tag": None,
            "sentiment": None
        }
    }]
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    repo_path = '.'  # Đường dẫn tới thư mục gốc dự án
    content = extract_readme(repo_path)
    if content:
        save_readme_to_json(content, '../collected_data/readme_content.json')
        print('README extracted.')
    else:
        print('README not found.')
