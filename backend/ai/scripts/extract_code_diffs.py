import subprocess
import json
import re
import os

def extract_code_diffs(repo_path, max_commits=100):
    os.chdir(repo_path)
    # Lấy danh sách commit hash
    hashes = subprocess.check_output(['git', 'rev-list', '--max-count', str(max_commits), 'HEAD'], encoding='utf-8').splitlines()
    diffs = []
    for h in hashes:
        diff = subprocess.check_output(['git', 'show', h, '--unified=0', '--no-color'], encoding='utf-8', errors='ignore')
        added = re.findall(r'^\+[^+][^\n]*', diff, re.MULTILINE)
        removed = re.findall(r'^-[^-][^\n]*', diff, re.MULTILINE)
        if added or removed:
            diffs.append({'commit': h, 'added': added, 'removed': removed})
    return diffs

def save_diffs_to_json(diffs, out_path, repo_name=None):
    output = []
    for idx, d in enumerate(diffs):
        # Lưu từng dòng added/removed như một mục riêng biệt
        for i, line in enumerate(d.get('added', [])):
            output.append({
                "id": f"diff_{d['commit']}_add_{i}",
                "data_type": "code_diff_text",
                "raw_text": line,
                "source_info": {
                    "repo_name": repo_name or "KLTN04",
                    "commit_sha": d['commit'],
                    "diff_type": "added"
                },
                "labels": {
                    "purpose": None,
                    "suspicious": None,
                    "tech_tag": None,
                    "sentiment": None
                }
            })
        for i, line in enumerate(d.get('removed', [])):
            output.append({
                "id": f"diff_{d['commit']}_rm_{i}",
                "data_type": "code_diff_text",
                "raw_text": line,
                "source_info": {
                    "repo_name": repo_name or "KLTN04",
                    "commit_sha": d['commit'],
                    "diff_type": "removed"
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
    repo_path = './backend'  # Đường dẫn repo backend (tương đối từ workspace)
    diffs = extract_code_diffs(repo_path)
    os.makedirs('../collected_data', exist_ok=True)
    save_diffs_to_json(diffs, '../collected_data/code_diffs.json')
    print(f'Extracted {len(diffs)} code diffs.')
