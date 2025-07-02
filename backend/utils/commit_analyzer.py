# backend/utils/commit_analyzer.py
"""
Commit analysis utility for extracting detailed information from commit data
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class CommitAnalyzer:
    """Analyzer for commit data to extract meaningful insights"""
    
    # File extension mappings for programming languages
    LANGUAGE_EXTENSIONS = {
        '.py': 'Python',
        '.js': 'JavaScript', 
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript React',
        '.jsx': 'JavaScript React',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.cs': 'C#',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.rs': 'Rust',
        '.sql': 'SQL',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.less': 'LESS',
        '.vue': 'Vue',
        '.json': 'JSON',
        '.xml': 'XML',
        '.yml': 'YAML',
        '.yaml': 'YAML',
        '.md': 'Markdown',
        '.txt': 'Text',
        '.sh': 'Shell',
        '.bat': 'Batch',
        '.ps1': 'PowerShell'
    }
    
    # Commit message patterns for change type detection
    CHANGE_TYPE_PATTERNS = {
        'feature': [r'\bfeat\b', r'\bfeature\b', r'\badd\b', r'\bimplement\b', r'\bnew\b'],
        'bugfix': [r'\bfix\b', r'\bbug\b', r'\brepair\b', r'\bresolve\b', r'\bcorrect\b'],
        'refactor': [r'\brefactor\b', r'\brestructure\b', r'\breorganize\b', r'\bcleanup\b'],
        'docs': [r'\bdoc\b', r'\bdocs\b', r'\bdocument\b', r'\breadme\b'],
        'test': [r'\btest\b', r'\btesting\b', r'\bspec\b'],
        'style': [r'\bstyle\b', r'\bformat\b', r'\blint\b', r'\bwhitespace\b'],
        'chore': [r'\bchore\b', r'\bmaintenance\b', r'\bupdate\b', r'\bbuild\b'],
        'performance': [r'\bperf\b', r'\bperformance\b', r'\boptimize\b', r'\bspeed\b']
    }
    
    @staticmethod
    def analyze_file_changes(modified_files: List[str]) -> Dict[str, Any]:
        """
        Analyze file changes to extract insights
        
        Args:
            modified_files: List of file paths that were modified
            
        Returns:
            Dictionary containing analysis results
        """
        if not modified_files:
            return {
                'file_types': {},
                'modified_directories': {},
                'languages': {},
                'file_categories': {}
            }
        
        file_types = {}
        modified_directories = {}
        languages = {}
        file_categories = {
            'source_code': 0,
            'configuration': 0,
            'documentation': 0,
            'assets': 0,
            'tests': 0
        }
        
        for file_path in modified_files:
            try:
                path = Path(file_path)
                
                # Extract file extension
                extension = path.suffix.lower()
                if extension:
                    file_types[extension] = file_types.get(extension, 0) + 1
                    
                    # Map to programming language
                    if extension in CommitAnalyzer.LANGUAGE_EXTENSIONS:
                        language = CommitAnalyzer.LANGUAGE_EXTENSIONS[extension]
                        languages[language] = languages.get(language, 0) + 1
                
                # Extract directory
                directory = str(path.parent) if path.parent != Path('.') else 'root'
                modified_directories[directory] = modified_directories.get(directory, 0) + 1
                
                # Categorize file type
                file_categories.update(CommitAnalyzer._categorize_file(file_path))
                
            except Exception as e:
                logger.warning(f"Error analyzing file {file_path}: {e}")
                continue
        
        return {
            'file_types': file_types,
            'modified_directories': modified_directories,
            'languages': languages,
            'file_categories': file_categories
        }
    
    @staticmethod
    def _categorize_file(file_path: str) -> Dict[str, int]:
        """Categorize a file into different types"""
        categories = {
            'source_code': 0,
            'configuration': 0,
            'documentation': 0,
            'assets': 0,
            'tests': 0
        }
        
        file_path_lower = file_path.lower()
        
        # Test files
        if any(keyword in file_path_lower for keyword in ['test', 'spec', '__test__', '.test.', '.spec.']):
            categories['tests'] = 1
        # Configuration files
        elif any(keyword in file_path_lower for keyword in [
            'config', 'setting', '.env', 'package.json', 'requirements.txt', 
            'dockerfile', 'docker-compose', 'makefile', '.yml', '.yaml'
        ]):
            categories['configuration'] = 1
        # Documentation
        elif any(keyword in file_path_lower for keyword in [
            'readme', '.md', 'doc', 'docs/', 'changelog', 'license'
        ]):
            categories['documentation'] = 1
        # Assets
        elif any(ext in file_path_lower for ext in [
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.css', '.scss', '.less'
        ]):
            categories['assets'] = 1
        # Source code (default for programming files)
        elif any(ext in file_path_lower for ext in [
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go'
        ]):
            categories['source_code'] = 1
        
        return categories
    
    @staticmethod
    def detect_change_type(commit_message: str) -> str:
        """
        Detect the type of change based on commit message
        
        Args:
            commit_message: The commit message to analyze
            
        Returns:
            Detected change type
        """
        message_lower = commit_message.lower()
        
        for change_type, patterns in CommitAnalyzer.CHANGE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return change_type
        
        return 'other'
    
    @staticmethod
    def categorize_commit_size(total_changes: int) -> str:
        """
        Categorize commit size based on total changes
        
        Args:
            total_changes: Total number of lines changed
            
        Returns:
            Size category (small, medium, large, massive)
        """
        if total_changes <= 10:
            return 'small'
        elif total_changes <= 50:
            return 'medium'
        elif total_changes <= 200:
            return 'large'
        else:
            return 'massive'
    
    @staticmethod
    def analyze_commit_pattern(commit_message: str) -> Dict[str, Any]:
        """
        Analyze commit message patterns for insights
        
        Args:
            commit_message: The commit message
            
        Returns:
            Dictionary with pattern analysis
        """
        analysis = {
            'has_conventional_format': False,
            'scope': None,
            'breaking_change': False,
            'sentiment': 'neutral',
            'urgency': 'normal'
        }
        
        # Check for conventional commit format (feat:, fix:, etc.)
        conventional_pattern = r'^(feat|fix|docs|style|refactor|test|chore|perf)(\(.+\))?!?:'
        if re.match(conventional_pattern, commit_message.lower()):
            analysis['has_conventional_format'] = True
            
            # Extract scope if present
            scope_match = re.search(r'\((.+)\)', commit_message)
            if scope_match:
                analysis['scope'] = scope_match.group(1)
        
        # Check for breaking changes
        if any(keyword in commit_message.lower() for keyword in [
            'breaking', 'breaking change', 'breaking:', '!:'
        ]):
            analysis['breaking_change'] = True
        
        # Simple sentiment analysis
        positive_words = ['improve', 'enhance', 'optimize', 'better', 'upgrade']
        negative_words = ['remove', 'delete', 'deprecated', 'broken', 'error']
        
        positive_count = sum(1 for word in positive_words if word in commit_message.lower())
        negative_count = sum(1 for word in negative_words if word in commit_message.lower())
        
        if positive_count > negative_count:
            analysis['sentiment'] = 'positive'
        elif negative_count > positive_count:
            analysis['sentiment'] = 'negative'
        
        # Check urgency indicators
        urgent_words = ['urgent', 'critical', 'hotfix', 'emergency', 'asap']
        if any(word in commit_message.lower() for word in urgent_words):
            analysis['urgency'] = 'high'
        
        return analysis
    
    @staticmethod
    def extract_commit_metadata(commit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and enrich commit metadata
        
        Args:
            commit_data: Raw commit data from GitHub API
            
        Returns:
            Enriched metadata dictionary
        """
        try:
            # Extract file changes if available
            modified_files = []
            if 'files' in commit_data:
                modified_files = [f['filename'] for f in commit_data['files']]
            
            # Analyze file changes
            file_analysis = CommitAnalyzer.analyze_file_changes(modified_files)
            
            # Calculate statistics
            additions = commit_data.get('stats', {}).get('additions', 0)
            deletions = commit_data.get('stats', {}).get('deletions', 0)
            total_changes = additions + deletions
            
            # Detect change type and size
            message = commit_data.get('commit', {}).get('message', '')
            change_type = CommitAnalyzer.detect_change_type(message)
            commit_size = CommitAnalyzer.categorize_commit_size(total_changes)
            
            # Analyze commit pattern
            pattern_analysis = CommitAnalyzer.analyze_commit_pattern(message)
            
            return {
                'modified_files': modified_files,
                'file_types': file_analysis['file_types'],
                'modified_directories': file_analysis['modified_directories'],
                'languages': file_analysis['languages'],
                'file_categories': file_analysis['file_categories'],
                'total_changes': total_changes,
                'change_type': change_type,
                'commit_size': commit_size,
                'pattern_analysis': pattern_analysis,
                'files_changed': len(modified_files),
                'insertions': additions,
                'deletions': deletions
            }
            
        except Exception as e:
            logger.error(f"Error extracting commit metadata: {e}")
            return {}
