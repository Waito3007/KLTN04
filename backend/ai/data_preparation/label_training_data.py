import json
import os
from typing import Dict, List, Any

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load raw JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def label_commit_data(commit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply labels to a commit entry."""
    # Extract commit message
    message = commit_data.get('raw_text', '').lower()
    
    # Initialize labels
    labels = {
        'purpose': categorize_purpose(message),
        'suspicious': detect_suspicious(message),
        'tech_tag': extract_tech_tags(message),
        'sentiment': analyze_sentiment(message)
    }
    
    commit_data['labels'] = labels
    return commit_data

def categorize_purpose(message: str) -> str:
    """Categorize the purpose of the commit with priority for keywords at the beginning."""
    message_lower = message.lower()

    # Define keyword categories
    feature_keywords = [
        'feat:', 'feature:', 'add', 'implement', 'new', 'create', 'support', 'enable', 'introduce',
        'develop', 'setup', 'start', 'initial', 'initialize', 'prepare', 'generate', 'build',
        'establish', 'integrate', 'incorporate', 'include', 'extend', 'expand', 'enhance',
        'upgrade', 'improve', 'optimize', 'update to', 'modernize', 'redesign', 'revamp',
        'restructure', 'reorganize', 'configure', 'setup', 'implement new', 'add support',
        'new feature', 'new functionality', 'new module', 'new component', 'new service',
        'new api', 'new endpoint', 'new interface', 'new design', 'new layout', 'new page',
        'new section', 'new option', 'new setting', 'new config', 'new validation', 'new test',
        'new requirement', 'new dependency', 'new library', 'new package', 'new plugin',
        'tính năng mới', 'thêm mới', 'hỗ trợ', 'giới thiệu', 'phát triển', 'tích hợp', 'mở rộng',
        'nâng cấp', 'cải thiện', 'tối ưu hóa', 'thiết kế mới', 'giao diện mới', 'chức năng mới'
    ]

    bugfix_keywords = [
        'fix', 'fix:', 'bug' 'bug:', 'resolve', 'issue', 'debug', 'troubleshoot', 'patch', 'correct',
        'solve', 'repair', 'handle', 'address', 'eliminate', 'prevent', 'avoid', 'mitigate',
        'workaround', 'hotfix', 'quickfix', 'bugfix', 'defect', 'problem', 'error', 'fail',
        'crash', 'exception', 'invalid', 'incorrect', 'wrong', 'break', 'broken', 'corrupt',
        'issue fix', 'bug report', 'error handling', 'exception handling', 'validation fix',
        'security fix', 'performance fix', 'memory leak', 'deadlock', 'race condition',
        'sửa lỗi', 'khắc phục', 'vá lỗi', 'xử lý lỗi', 'giải quyết vấn đề', 'lỗi bảo mật',
        'lỗi hiệu năng', 'lỗi bộ nhớ', 'lỗi điều kiện'
    ]

    documentation_keywords = [
        'doc:', 'docs:', 'document', 'readme', 'documentation', 'comment', 'describe',
        'explain', 'clarify', 'note', 'guide', 'manual', 'instruction', 'reference',
        'example', 'sample', 'template', 'demonstration', 'tutorial', 'howto', 'wiki',
        'changelog', 'version', 'release note', 'api doc', 'javadoc', 'docstring',
        'annotation', 'specification', 'requirement doc', 'design doc', 'architecture doc',
        'user guide', 'developer guide', 'installation guide', 'deployment guide',
        'configuration guide', 'usage guide', 'api reference', 'code example',
        'tài liệu', 'hướng dẫn', 'ghi chú', 'mô tả', 'giải thích', 'chú thích', 'ví dụ',
        'mẫu', 'hướng dẫn sử dụng', 'hướng dẫn cài đặt', 'hướng dẫn triển khai'
    ]

    refactoring_keywords = [
        'refactor:', 'clean:', 'reorganize', 'restructure', 'rewrite', 'rework', 'improve',
        'simplify', 'optimize', 'enhance', 'modernize', 'standardize', 'normalize',
        'format', 'style', 'lint', 'cleanup', 'polish', 'revise', 'update', 'upgrade',
        'maintain', 'organize', 'refine', 'streamline', 'consolidate', 'unify', 'merge',
        'split', 'extract', 'separate', 'modularize', 'decompose', 'abstract', 'encapsulate',
        'tái cấu trúc', 'làm sạch', 'tối ưu', 'chuẩn hóa', 'định dạng', 'cải tiến', 'sắp xếp'
    ]

    test_keywords = [
        'test:', 'testing:', 'unittest', 'spec:', 'tdd:', 'e2e:', 'integration test',
        'unit test', 'regression test', 'coverage', 'benchmark test', 'performance test',
        'load test', 'stress test', 'mock', 'stub', 'assertion', 'verify', 'validate', 'check',
        'kiểm thử', 'kiểm tra', 'test hiệu năng', 'test tích hợp', 'test đơn vị', 'xác minh'
    ]

    chore_keywords = [
        'chore:', 'build:', 'ci:', 'task:', 'dependency', 'package', 'config', 'tooling',
        'pipeline', 'workflow', 'automation', 'script', 'task', 'housekeeping', 'maintenance',
        'cleanup', 'setup', 'upgrade', 'version bump', 'release', 'tag', 'publish', 'deploy',
        'công việc', 'tự động hóa', 'quy trình', 'cấu hình', 'phụ thuộc', 'phát hành', 'triển khai'
    ]

    style_keywords = [
        'style:', 'format:', 'lint:', 'prettify:', 'beautify:', 'indent:', 'formatting',
        'whitespace', 'indentation', 'semicolon', 'quotes', 'coding style', 'eslint', 'prettier',
        'định dạng', 'kiểu mã hóa', 'làm đẹp', 'thụt lề', 'dấu chấm phẩy', 'dấu ngoặc kép'
    ]

    # Check for keywords at the beginning of the message
    first_word = message.split()[0] if message else ''
    if first_word in feature_keywords:
        return 'Feature Implementation'
    elif first_word in bugfix_keywords:
        return 'Bug Fix'
    elif first_word in documentation_keywords:
        return 'Documentation Update'
    elif first_word in refactoring_keywords:
        return 'Code Refactoring'
    elif first_word in test_keywords:
        return 'Test'
    elif first_word in chore_keywords:
        return 'Chore'
    elif first_word in style_keywords:
        return 'Style'

    # Fallback to checking entire message
    for keyword in feature_keywords:
        if keyword in message_lower:
            return 'Feature Implementation'
    for keyword in bugfix_keywords:
        if keyword in message_lower:
            return 'Bug Fix'
    for keyword in documentation_keywords:
        if keyword in message_lower:
            return 'Documentation Update'
    for keyword in refactoring_keywords:
        if keyword in message_lower:
            return 'Code Refactoring'
    for keyword in test_keywords:
        if keyword in message_lower:
            return 'Test'
    for keyword in chore_keywords:
        if keyword in message_lower:
            return 'Chore'
    for keyword in style_keywords:
        if keyword in message_lower:
            return 'Style'

    return 'Other'

def detect_suspicious(message: str) -> int:
    """Detect if commit message indicates suspicious changes."""
    suspicious_patterns = [
        'hack', 'temporary', 'workaround', 'quick fix',
        'remove security', 'bypass', 'backdoor'
    ]
    return 1 if any(pattern in message for pattern in suspicious_patterns) else 0

def extract_tech_tags(message: str) -> List[str]:
    """Extract technology tags from commit message."""
    tech_tags = []
    tech_patterns = {
        'react': ['react', 'jsx', 'component'],
        'typescript': ['typescript', 'ts'],
        'python': ['python', 'py'],
        'docker': ['docker', 'container'],
        'database': ['sql', 'database', 'migration'],
        'api': ['api', 'endpoint', 'rest'],
        'css': ['css', 'style', 'scss'],
        'github': ['github', 'git'],
        'ci': ['ci', 'pipeline', 'workflow']
    }
    
    for tech, patterns in tech_patterns.items():
        if any(pattern in message for pattern in patterns):
            tech_tags.append(tech)
    
    return tech_tags

def analyze_sentiment(message: str) -> str:
    """Analyze sentiment of commit message."""
    positive_words = ['improve', 'enhance', 'optimize', 'better', 'fix', 'success']
    negative_words = ['break', 'crash', 'error', 'fail', 'bug', 'issue']
    
    if any(word in message for word in positive_words):
        return 'positive'
    elif any(word in message for word in negative_words):
        return 'negative'
    else:
        return 'neutral'

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'collected_data', 'commit_messages_raw.json')
    output_path = os.path.join(base_dir, 'training_data', 'han_training_samples.json')
    
    # Load raw data
    data = load_json_data(input_path)
    
    # Apply labels
    labeled_data = [label_commit_data(commit) for commit in data]
    
    # Save labeled data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=2)
    
    print(f"Labeled data saved to {output_path}")

if __name__ == "__main__":
    main()
