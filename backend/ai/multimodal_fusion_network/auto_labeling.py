import json
import os
import re
import logging
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from pathlib import Path
from collections import Counter

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("auto_labeling.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tải các model cần thiết
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Biến toàn cục để kiểm tra xem spaCy có sẵn không
USE_SPACY = False
nlp = None

# Chỉ thử import spaCy khi thực sự cần
try:
    import importlib
    spacy_spec = importlib.util.find_spec("spacy")
    if spacy_spec is not None:
        spacy = importlib.import_module("spacy")
        try:
            nlp = spacy.load("en_core_web_sm")
            USE_SPACY = True
            logger.info("spaCy loaded successfully")
        except Exception as e:
            logger.warning(f"spaCy found but cannot load model: {str(e)}. Using NLTK only.")
    else:
        logger.warning("spaCy not installed. Using NLTK only.")
except Exception as e:
    logger.warning(f"spaCy import failed: {str(e)}. Using NLTK only.")

# Các từ khóa cho phân loại
KEYWORDS = {
    "task_type": {
        "development": ["feature", "implement", "add", "create", "develop", "support", "enhancement"],
        "bug_fix": ["fix", "bug", "issue", "problem", "error", "crash", "resolve", "patch"],
        "refactoring": ["refactor", "clean", "reorganize", "restructure", "simplify", "optimize", "improve"],
        "documentation": ["doc", "documentation", "comment", "readme", "guide", "tutorial"],
        "testing": ["test", "unittest", "spec", "coverage", "integration test"],
        "devops": ["deploy", "ci", "cd", "pipeline", "build", "release", "docker", "kubernetes"],
        "security": ["security", "vulnerability", "auth", "authentication", "authorization", "encrypt"]
    },
    "complexity": {
        "low": ["simple", "easy", "minor", "small", "typo", "quick"],
        "medium": ["moderate", "several", "multiple", "update"],
        "high": ["complex", "major", "significant", "difficult", "challenging", "rewrite", "redesign"]
    },
    "technical_area": {
        "frontend": ["ui", "ux", "interface", "css", "html", "responsive", "component", "react", "angular", "vue"],
        "backend": ["api", "endpoint", "server", "controller", "service", "middleware"],
        "database": ["database", "db", "sql", "query", "schema", "migration", "model"],
        "infrastructure": ["infra", "config", "setup", "environment", "server", "cloud", "aws", "azure", "gcp"],
        "mobile": ["android", "ios", "mobile", "app", "responsive"]
    },
    "required_skills": {
        "python": ["python", "django", "flask", "pandas", "numpy", "pytorch", "tensorflow"],
        "javascript": ["javascript", "js", "node", "npm", "yarn", "webpack"],
        "react": ["react", "jsx", "component", "hook", "redux", "state"],
        "angular": ["angular", "ng", "typescript", "component"],
        "vue": ["vue", "vuex", "nuxt", "quasar"],
        "java": ["java", "spring", "hibernate", "maven", "gradle"],
        "dotnet": ["c#", ".net", "asp.net", "core", "entity framework"],
        "database": ["sql", "mysql", "postgresql", "mongodb", "nosql", "database"],
        "devops": ["docker", "kubernetes", "k8s", "ci/cd", "jenkins", "github actions", "travis"],
        "testing": ["test", "jest", "mocha", "cypress", "selenium", "unit test"]
    },
    "priority": {
        "critical": ["urgent", "critical", "emergency", "asap", "highest", "blocker", "immediately"],
        "high": ["high", "important", "priority", "significant"],
        "medium": ["medium", "moderate", "normal"],
        "low": ["low", "minor", "trivial", "whenever", "can wait"]
    }
}

# File extensions to technical area mapping
FILE_EXTENSIONS = {
    "frontend": [".js", ".jsx", ".ts", ".tsx", ".vue", ".css", ".scss", ".html", ".svg"],
    "backend": [".py", ".rb", ".php", ".java", ".go", ".rs", ".cs", ".js", ".ts"],
    "database": [".sql", ".prisma", ".graphql", ".gql"],
    "mobile": [".swift", ".kt", ".java", ".dart", ".m", ".h"],
    "devops": [".yml", ".yaml", ".toml", ".tf", ".dockerfile", "Dockerfile", ".ini", ".conf"]
}

# File paths to technical area mapping
PATH_PATTERNS = {
    "frontend": [r"(front|ui|client|web|app)", r"(components|pages|views|screens)"],
    "backend": [r"(back|api|server|service)", r"(controllers|models|middleware|routes)"],
    "database": [r"(db|database|migration|schema|seed)", r"(models|entities)"],
    "infrastructure": [r"(infra|deploy|config)", r"(terraform|k8s|docker)"],
    "testing": [r"(test|spec|__tests__)", r"(unit|integration|e2e)"]
}

def clean_text(text):
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keeping spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_message(commit):
    """Extract commit message from a commit object."""
    if not commit:
        return ""
    
    # If the commit is from the GitHub API format
    if "commit" in commit and "message" in commit["commit"]:
        return commit["commit"]["message"]
    
    # If it's directly a message field
    if "message" in commit:
        return commit["message"]
    
    # If there's a text field (common in some dataset formats)
    if "text" in commit:
        return commit["text"]
    
    return ""

def extract_changed_files(commit):
    """Extract the list of files changed in this commit."""
    files = []
    
    # GitHub API format
    if "files" in commit:
        for file_info in commit["files"]:
            if "filename" in file_info:
                files.append(file_info["filename"])
    
    return files

def get_label_scores(text, changed_files, label_type):
    """Calculate scores for each label in the given label type."""
    if label_type not in KEYWORDS:
        return {}
    
    # Clean and tokenize text
    clean_msg = clean_text(text)
    tokens = word_tokenize(clean_msg)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    # Initialize scores
    scores = {label: 0 for label in KEYWORDS[label_type].keys()}
    
    # Sử dụng spaCy nếu có, nếu không thì sử dụng chỉ NLTK
    if USE_SPACY and nlp is not None:
        # Process with spaCy for better tokenization and lemmatization
        doc = nlp(clean_msg)
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Score based on keywords in message
        for label, keywords in KEYWORDS[label_type].items():
            # Count keyword matches in tokens and lemmas
            token_matches = sum(1 for token in tokens if any(keyword == token or keyword in token for keyword in keywords))
            lemma_matches = sum(1 for lemma in lemmas if any(keyword == lemma or keyword in lemma for keyword in keywords))
            
            # Apply logarithmic scaling to avoid extreme scores
            scores[label] = np.log1p(token_matches + lemma_matches) if (token_matches + lemma_matches) > 0 else 0
    else:
        # Chỉ sử dụng NLTK tokens (không có lemmatization)
        for label, keywords in KEYWORDS[label_type].items():
            # Count keyword matches in tokens
            token_matches = sum(1 for token in tokens if any(keyword == token or keyword in token for keyword in keywords))
            # Count whole phrase matches
            phrase_matches = sum(1 for keyword in keywords if keyword in clean_msg)
            
            # Apply logarithmic scaling to avoid extreme scores
            scores[label] = np.log1p(token_matches + phrase_matches) if (token_matches + phrase_matches) > 0 else 0
    
    # If scoring technical area, also consider file extensions and paths
    if label_type == "technical_area" and changed_files:
        for file_path in changed_files:
            # Get extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Score based on file extension
            for area, extensions in FILE_EXTENSIONS.items():
                if ext in extensions:
                    scores[area] = scores.get(area, 0) + 0.5
            
            # Score based on path patterns
            for area, patterns in PATH_PATTERNS.items():
                if any(re.search(pattern, file_path, re.IGNORECASE) for pattern in patterns):
                    scores[area] = scores.get(area, 0) + 0.5
    
    # Similarly, adjust required_skills based on file extensions
    if label_type == "required_skills" and changed_files:
        for file_path in changed_files:
            ext = os.path.splitext(file_path)[1].lower()
            
            # Map extensions to skills
            if ext in [".py"]:
                scores["python"] = scores.get("python", 0) + 0.5
            elif ext in [".js", ".jsx"]:
                scores["javascript"] = scores.get("javascript", 0) + 0.5
            elif ext in [".ts", ".tsx"]:
                scores["javascript"] = scores.get("javascript", 0) + 0.3
                scores["angular"] = scores.get("angular", 0) + 0.2
            elif ext in [".jsx", ".tsx"] and "react" in file_path.lower():
                scores["react"] = scores.get("react", 0) + 0.5
            elif ext in [".vue"]:
                scores["vue"] = scores.get("vue", 0) + 0.5
            elif ext in [".java"]:
                scores["java"] = scores.get("java", 0) + 0.5
            elif ext in [".cs"]:
                scores["dotnet"] = scores.get("dotnet", 0) + 0.5
            elif ext in [".sql"]:
                scores["database"] = scores.get("database", 0) + 0.5
    
    return scores

def get_top_labels(scores, threshold=0.0, max_labels=2):
    """Get top labels based on scores, applying a minimum threshold."""
    if not scores:
        return []
    
    # Sort labels by score in descending order
    sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get labels above threshold, up to max_labels
    top_labels = [label for label, score in sorted_labels if score > threshold][:max_labels]
    
    return top_labels

def label_commit(commit):
    """Generate labels for a commit."""
    message = extract_message(commit)
    changed_files = extract_changed_files(commit)
    
    if not message:
        return {}
    
    # Calculate scores for each label type
    task_type_scores = get_label_scores(message, changed_files, "task_type")
    complexity_scores = get_label_scores(message, changed_files, "complexity")
    technical_area_scores = get_label_scores(message, changed_files, "technical_area")
    required_skills_scores = get_label_scores(message, changed_files, "required_skills")
    priority_scores = get_label_scores(message, changed_files, "priority")
    
    # Determine top labels for each type
    task_types = get_top_labels(task_type_scores, threshold=0.1, max_labels=2)
    complexity = get_top_labels(complexity_scores, threshold=0.1, max_labels=1)
    technical_areas = get_top_labels(technical_area_scores, threshold=0.1, max_labels=2)
    required_skills = get_top_labels(required_skills_scores, threshold=0.1, max_labels=3)
    priority = get_top_labels(priority_scores, threshold=0.1, max_labels=1)
    
    # If any category doesn't have labels above threshold, use the highest scoring one
    if not task_types and task_type_scores:
        task_types = [max(task_type_scores.items(), key=lambda x: x[1])[0]]
    
    if not complexity and complexity_scores:
        complexity = [max(complexity_scores.items(), key=lambda x: x[1])[0]]
    
    if not technical_areas and technical_area_scores:
        technical_areas = [max(technical_area_scores.items(), key=lambda x: x[1])[0]]
    
    if not required_skills and required_skills_scores:
        required_skills = [max(required_skills_scores.items(), key=lambda x: x[1])[0]]
    
    if not priority and priority_scores:
        priority = [max(priority_scores.items(), key=lambda x: x[1])[0]]
    
    # Default values for empty categories
    if not task_types:
        task_types = ["development"]  # Default task type
    
    if not complexity:
        complexity = ["medium"]  # Default complexity
    
    if not priority:
        priority = ["medium"]  # Default priority
    
    # Create labels object
    labels = {
        "task_type": task_types,
        "complexity": complexity[0] if complexity else "medium",
        "technical_area": technical_areas,
        "required_skills": required_skills,
        "priority": priority[0] if priority else "medium"
    }
    
    return labels

def process_commit_file(input_file, output_file):
    """Process a file of commits, adding labels to each commit."""
    logger.info(f"Processing file: {input_file}")
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract commits
        commits = data.get('data', [])
        if not commits:
            logger.warning(f"No commits found in {input_file}")
            return
        
        logger.info(f"Found {len(commits)} commits to process")
        
        # Process each commit
        labeled_commits = []
        for i, commit in enumerate(commits):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(commits)} commits...")
            
            # Skip empty commits
            message = extract_message(commit)
            if not message:
                continue
            
            # Generate labels
            labels = label_commit(commit)
            
            # Add labels to commit
            commit_copy = commit.copy()
            commit_copy['labels'] = labels
            labeled_commits.append(commit_copy)
        
        # Update metadata
        metadata = data.get('metadata', {}).copy()
        metadata['total_samples'] = len(labeled_commits)
        metadata['labeled_at'] = datetime.now().isoformat()
        metadata['labeling_version'] = '1.0'
        
        # Create output data
        output_data = {
            'metadata': metadata,
            'data': labeled_commits
        }
        
        # Write output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Labeled {len(labeled_commits)} commits and saved to {output_file}")
        
        # Generate label statistics
        generate_label_stats(labeled_commits, os.path.join(os.path.dirname(output_file), "label_stats.json"))
        
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        raise

def generate_label_stats(labeled_commits, output_file):
    """Generate statistics about the labels assigned to commits."""
    stats = {
        "total_commits": len(labeled_commits),
        "label_counts": {
            "task_type": Counter(),
            "complexity": Counter(),
            "technical_area": Counter(),
            "required_skills": Counter(),
            "priority": Counter()
        }
    }
    
    # Count labels
    for commit in labeled_commits:
        labels = commit.get('labels', {})
        
        # Count task types (can be multiple)
        for task_type in labels.get('task_type', []):
            stats["label_counts"]["task_type"][task_type] += 1
        
        # Count complexity (single value)
        complexity = labels.get('complexity')
        if complexity:
            stats["label_counts"]["complexity"][complexity] += 1
        
        # Count technical areas (can be multiple)
        for area in labels.get('technical_area', []):
            stats["label_counts"]["technical_area"][area] += 1
        
        # Count required skills (can be multiple)
        for skill in labels.get('required_skills', []):
            stats["label_counts"]["required_skills"][skill] += 1
        
        # Count priority (single value)
        priority = labels.get('priority')
        if priority:
            stats["label_counts"]["priority"][priority] += 1
    
    # Convert Counters to dictionaries for JSON serialization
    for category in stats["label_counts"]:
        stats["label_counts"][category] = dict(stats["label_counts"][category])
    
    # Save statistics
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Generated label statistics and saved to {output_file}")

def main():
    """Main function to process commit files and add labels."""
    import argparse
    
    # Tạo parser để xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Gắn nhãn tự động cho commit datasets")
    parser.add_argument("--input", "-i", help="Đường dẫn đến file JSON chứa commit cần gán nhãn", type=str)
    parser.add_argument("--output", "-o", help="Đường dẫn đến file JSON đầu ra (tùy chọn)", type=str)
    parser.add_argument("--output-dir", "-d", help="Thư mục đầu ra cho file đã gán nhãn (mặc định: ./data/labeled)", type=str)
    
    args = parser.parse_args()
    
    # Define input and output directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir if args.output_dir else os.path.join(base_dir, "data", "labeled")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Nếu không có tham số input, sử dụng thư mục mặc định
    if not args.input:
        input_dir = os.path.join(base_dir, "data", "processed")
        # Find all JSON files in the input directory
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.json') and 'merged_commits' in f]
        
        if not input_files:
            logger.warning(f"No merged commit files found in {input_dir}")
            return
        
        logger.info(f"Found {len(input_files)} files to process: {', '.join(input_files)}")
        
        # Process each file
        for input_file in input_files:
            input_path = os.path.join(input_dir, input_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"labeled_commits_{timestamp}.json"
            output_path = os.path.join(output_dir, output_file)
            
            logger.info(f"Processing {input_file} -> {output_file}")
            process_commit_file(input_path, output_path)
    else:
        # Sử dụng file cụ thể được chỉ định
        input_path = args.input
        if not os.path.exists(input_path):
            logger.error(f"Input file does not exist: {input_path}")
            return
        
        # Xác định file đầu ra
        if args.output:
            output_path = args.output
        else:
            # Tạo tên file đầu ra dựa trên tên file đầu vào
            input_filename = os.path.basename(input_path)
            base_name = os.path.splitext(input_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_labeled_{timestamp}.json"
            output_path = os.path.join(output_dir, output_file)
        
        logger.info(f"Processing single file: {input_path} -> {output_path}")
        process_commit_file(input_path, output_path)
    
    logger.info("All files processed successfully!")

if __name__ == "__main__":
    main()