import json
import re
from pathlib import Path
from typing import List, Dict

class CommitFilter:
    def __init__(self):
        self.commit_types = {
            'refactor': [
                r'refactor',
                r'restructure',
                r'reorganize',
                r'cleanup',
                r'remove unused',
                r'simplify',
                r'optimize',
                r'refine'
            ],
            'documentation': [
                r'docs?:?',
                r'document',
                r'update readme',
                r'improve description',
                r'add example',
                r'fix typo',
                r'guide',
                r'manual'
            ],
            'test': [
                r'test:?',
                r'testing',
                r'coverage',
                r'unit test',
                r'integration test',
                r'e2e test',
                r'verify',
                r'add coverage'
            ],
            'chore': [
                r'chore:?',
                r'build:?',
                r'ci:?',
                r'update dependencies',
                r'bump version',
                r'release',
                r'maintenance',
                r'workflow',
                r'actions'
            ]
        }
        
        # Compile regex patterns
        self.patterns = {
            commit_type: [re.compile(pattern, re.IGNORECASE) 
                         for pattern in patterns]
            for commit_type, patterns in self.commit_types.items()
        }

    def classify_commit(self, commit_message: str) -> str:
        """Classify a commit message into one of the defined types."""
        commit_message = commit_message.lower()
        
        for commit_type, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(commit_message):
                    return commit_type
        
        return None  # Return None if no match found

    def filter_commits(self, commits: List[Dict]) -> Dict[str, List[Dict]]:
        """Filter commits into different types."""
        filtered_commits = {
            'refactor': [],
            'documentation': [],
            'test': [],
            'chore': []
        }
        
        ignored_commits = []
        
        for commit in commits:
            commit_type = self.classify_commit(commit['raw_text'])
            if commit_type:
                filtered_commits[commit_type].append(commit)
            else:
                ignored_commits.append(commit)
                
        return filtered_commits, ignored_commits

    def analyze_commit_distribution(self, filtered_commits: Dict[str, List[Dict]]):
        """Analyze the distribution of commit types."""
        distribution = {
            commit_type: len(commits)
            for commit_type, commits in filtered_commits.items()
        }
        return distribution

def main():
    # Load commit data from training files
    base_dir = Path(__file__).parent.parent
    print(f"Base directory: {base_dir}")
    data_files = [
        base_dir / "collected_data" / "commit_messages_raw.json"
    ]
    
    print("Looking for files:")
    for file_path in data_files:
        print(f"- {file_path} {'(exists)' if file_path.exists() else '(not found)'}")
    
    all_commits = []
    for file_path in data_files:
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    commits = json.load(f)
                    all_commits.extend(commits)
                print(f"Loaded {len(commits)} commits from {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {str(e)}")
    
    if not all_commits:
        print("No commits loaded!")
        return
        
    print(f"\nTotal commits loaded: {len(all_commits)}")
    
    # Filter commits
    commit_filter = CommitFilter()
    filtered_commits, ignored = commit_filter.filter_commits(all_commits)
    
    # Analyze distribution
    distribution = commit_filter.analyze_commit_distribution(filtered_commits)
    
    print("\nCommit Type Distribution:")
    for commit_type, count in distribution.items():
        print(f"{commit_type}: {count} commits")
    
    print(f"\nIgnored commits: {len(ignored)}")
    
    # Save filtered commits
    output_dir = base_dir / "collected_data"
    output_dir.mkdir(exist_ok=True)
    
    for commit_type, commits in filtered_commits.items():
        if commits:  # Only save if there are commits of this type
            output_file = output_dir / f"filtered_{commit_type}_commits.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(commits, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {len(commits)} {commit_type} commits to {output_file.name}")
    
    # Save ignored commits for review
    if ignored:
        output_file = output_dir / "ignored_commits.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ignored, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(ignored)} ignored commits to {output_file.name}")

if __name__ == "__main__":
    main()
