"""
Data Generation for Multi-Modal Fusion Network
Tạo synthetic GitHub commit data với realistic patterns
"""

import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import string
from collections import defaultdict
import re


class GitHubDataGenerator:
    """
    Generator cho synthetic GitHub commit data với metadata patterns
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Pre-defined patterns
        self.commit_patterns = self._init_commit_patterns()
        self.file_types = self._init_file_types()
        self.authors = self._init_authors()
        self.programming_words = self._init_programming_words()
        
    def _init_commit_patterns(self) -> Dict[str, List[str]]:
        """Initialize commit message patterns"""
        return {
            'fix': [
                "Fix bug in {component}",
                "Fixed {issue} causing {problem}",
                "Bugfix: {description}",
                "Resolve {issue} in {component}",
                "Patch for {vulnerability}",
                "Hotfix: {critical_issue}",
                "Quick fix for {problem}",
                "Emergency fix: {description}"
            ],
            'feature': [
                "Add {feature} to {component}",
                "Implement {functionality}",
                "Feature: {new_feature}",
                "Introduce {capability}",
                "New: {feature_description}",
                "Enhance {component} with {feature}",
                "Added support for {technology}",
                "Initial implementation of {feature}"
            ],
            'refactor': [
                "Refactor {component} for better {quality}",
                "Code cleanup in {module}",
                "Restructure {component}",
                "Optimize {algorithm} implementation",
                "Improve {aspect} of {component}",
                "Reorganize {module} structure",
                "Clean up {technical_debt}",
                "Modernize {legacy_code}"
            ],
            'update': [
                "Update {dependency} to version {version}",
                "Upgrade {library} dependencies",
                "Bump {package} version",
                "Update documentation for {feature}",
                "Sync with upstream {repository}",
                "Update configuration for {environment}",
                "Refresh {cache} implementation",
                "Update {api} to latest version"
            ],
            'test': [
                "Add tests for {component}",
                "Test coverage for {feature}",
                "Unit tests for {module}",
                "Integration tests for {system}",
                "Fix failing tests in {component}",
                "Improve test stability",
                "Add regression tests for {bug}",
                "Update test data for {scenario}"
            ]
        }
    
    def _init_file_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize file type patterns"""
        return {
            'python': {
                'extensions': ['.py', '.pyx', '.pyi'],
                'risk_factor': 0.7,
                'complexity_base': 0.6
            },
            'javascript': {
                'extensions': ['.js', '.jsx', '.ts', '.tsx'],
                'risk_factor': 0.8,
                'complexity_base': 0.7
            },
            'java': {
                'extensions': ['.java', '.kt', '.scala'],
                'risk_factor': 0.6,
                'complexity_base': 0.5
            },
            'cpp': {
                'extensions': ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp'],
                'risk_factor': 0.9,
                'complexity_base': 0.8
            },
            'config': {
                'extensions': ['.json', '.yml', '.yaml', '.xml', '.toml', '.ini'],
                'risk_factor': 0.3,
                'complexity_base': 0.2
            },
            'documentation': {
                'extensions': ['.md', '.rst', '.txt', '.doc'],
                'risk_factor': 0.1,
                'complexity_base': 0.1
            },
            'web': {
                'extensions': ['.html', '.css', '.scss', '.less'],
                'risk_factor': 0.4,
                'complexity_base': 0.3
            },
            'database': {
                'extensions': ['.sql', '.psql', '.mysql'],
                'risk_factor': 0.7,
                'complexity_base': 0.6
            }
        }
    
    def _init_authors(self) -> List[Dict[str, Any]]:
        """Initialize author profiles"""
        return [
            {'name': 'senior_dev_1', 'experience': 0.9, 'reliability': 0.95, 'activity': 0.8},
            {'name': 'senior_dev_2', 'experience': 0.85, 'reliability': 0.9, 'activity': 0.7},
            {'name': 'mid_dev_1', 'experience': 0.6, 'reliability': 0.8, 'activity': 0.9},
            {'name': 'mid_dev_2', 'experience': 0.65, 'reliability': 0.75, 'activity': 0.85},
            {'name': 'mid_dev_3', 'experience': 0.7, 'reliability': 0.82, 'activity': 0.8},
            {'name': 'junior_dev_1', 'experience': 0.3, 'reliability': 0.6, 'activity': 0.95},
            {'name': 'junior_dev_2', 'experience': 0.25, 'reliability': 0.65, 'activity': 0.9},
            {'name': 'junior_dev_3', 'experience': 0.35, 'reliability': 0.7, 'activity': 0.88},
            {'name': 'intern_1', 'experience': 0.1, 'reliability': 0.5, 'activity': 0.7},
            {'name': 'intern_2', 'experience': 0.15, 'reliability': 0.55, 'activity': 0.75}
        ]
    
    def _init_programming_words(self) -> Dict[str, List[str]]:
        """Initialize programming-related words"""
        return {
            'components': [
                'API', 'database', 'frontend', 'backend', 'service', 'module', 'controller',
                'model', 'view', 'router', 'middleware', 'authentication', 'authorization',
                'cache', 'session', 'webhook', 'scheduler', 'queue', 'worker', 'parser',
                'validator', 'serializer', 'repository', 'factory', 'adapter', 'connector'
            ],
            'issues': [
                'memory leak', 'race condition', 'deadlock', 'null pointer', 'buffer overflow',
                'security vulnerability', 'performance issue', 'timeout', 'connection error',
                'validation error', 'parsing error', 'encoding issue', 'permission denied',
                'resource exhaustion', 'infinite loop', 'stack overflow', 'dependency conflict'
            ],
            'features': [
                'real-time notifications', 'user dashboard', 'data analytics', 'file upload',
                'search functionality', 'user authentication', 'payment processing',
                'email integration', 'social login', 'API rate limiting', 'data export',
                'mobile responsiveness', 'dark mode', 'internationalization', 'audit logs'
            ],
            'technologies': [
                'Docker', 'Kubernetes', 'Redis', 'PostgreSQL', 'MongoDB', 'Elasticsearch',
                'RabbitMQ', 'Kafka', 'GraphQL', 'REST API', 'gRPC', 'WebSocket', 'OAuth',
                'JWT', 'TLS', 'HTTPS', 'AWS', 'Azure', 'GCP', 'Terraform', 'Ansible'
            ]
        }
    
    def generate_commit_message(self, commit_type: str, risk_level: float) -> str:
        """
        Generate realistic commit message
        
        Args:
            commit_type: Type of commit (fix, feature, etc.)
            risk_level: Risk level (0-1) to influence message complexity
            
        Returns:
            Generated commit message
        """
        if commit_type not in self.commit_patterns:
            commit_type = random.choice(list(self.commit_patterns.keys()))
        
        template = random.choice(self.commit_patterns[commit_type])
        
        # Fill template with appropriate words
        component = random.choice(self.programming_words['components'])
        issue = random.choice(self.programming_words['issues'])
        feature = random.choice(self.programming_words['features'])
        technology = random.choice(self.programming_words['technologies'])
        
        # Replace placeholders
        message = template.format(
            component=component,
            issue=issue,
            problem=issue,
            feature=feature,
            functionality=feature,
            new_feature=feature,
            capability=feature,
            feature_description=feature,
            technology=technology,
            quality='performance' if risk_level > 0.5 else 'maintainability',
            module=component,
            aspect='security' if risk_level > 0.7 else 'performance',
            technical_debt='legacy code',
            legacy_code='deprecated functions',
            dependency=technology,
            version=f"{random.randint(1,5)}.{random.randint(0,20)}.{random.randint(0,10)}",
            library=technology,
            package=technology,
            environment='production' if risk_level > 0.6 else 'development',
            repository='main',
            cache='Redis',
            api='REST API',
            system=component,
            bug=issue,
            scenario='edge case',
            description=issue,
            vulnerability='SQL injection',
            critical_issue='server crash',
            algorithm='sorting'
        )
          # Add complexity based on risk level
        if risk_level > 0.8:
            suffixes = [
                " - critical security patch",
                " - urgent production fix",
                " - breaking change",
                " - requires migration",
                " - affects multiple services"
            ]
            message += random.choice(suffixes)
        elif risk_level > 0.6:
            suffixes = [
                " - needs testing",
                " - requires review",
                " - performance impact",
                " - config change needed"
            ]
            message += random.choice(suffixes)
        
        return message
    
    def generate_file_changes(self, risk_level: float, complexity_level: float) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Generate realistic file changes
        
        Args:
            risk_level: Risk level (0-1)
            complexity_level: Complexity level (0-1)
            
        Returns:
            Tuple of (file_data_list, change_stats)
        """
        # Determine number of files based on complexity
        if complexity_level > 0.8:
            num_files = random.randint(8, 25)
        elif complexity_level > 0.6:
            num_files = random.randint(4, 12)
        elif complexity_level > 0.3:
            num_files = random.randint(2, 6)
        else:
            num_files = random.randint(1, 3)
        
        files_data = []
        stats = {'additions': 0, 'deletions': 0, 'modifications': 0}
        
        # Select file types based on risk level
        type_weights = []
        for file_type, info in self.file_types.items():
            weight = info['risk_factor'] if risk_level > 0.5 else (1 - info['risk_factor'])
            type_weights.append((file_type, weight, info))
        
        for i in range(num_files):
            # Select file type
            selected_type = random.choices(
                [t[0] for t in type_weights],
                weights=[t[1] for t in type_weights]
            )[0]
            
            type_info = self.file_types[selected_type]
            extension = random.choice(type_info['extensions'])
            
            # Generate file path
            components = random.choice(self.programming_words['components']).lower()
            file_name = f"{components}_{random.randint(1, 100)}{extension}"
            
            # Create realistic directory structure
            if selected_type == 'python':
                dirs = ['src', 'lib', 'api', 'models', 'views', 'utils']
            elif selected_type == 'javascript':
                dirs = ['src', 'components', 'pages', 'utils', 'services']
            elif selected_type == 'java':
                dirs = ['src/main/java', 'src/test/java']
            elif selected_type == 'config':
                dirs = ['config', 'deploy', 'scripts']
            elif selected_type == 'documentation':
                dirs = ['docs', 'README']
            else:
                dirs = ['src', 'lib', 'assets']
            
            directory = random.choice(dirs)
            file_path = f"{directory}/{file_name}"
            
            # Generate change statistics for this file
            base_changes = int(complexity_level * 200)
            additions = random.randint(1, max(1, base_changes))
            deletions = random.randint(0, max(1, int(additions * 0.7)))
            changes = additions + deletions
            
            # Create file data dict as expected by MetadataProcessor
            file_data = {
                'filename': file_path,
                'additions': additions,
                'deletions': deletions,
                'changes': changes,
                'status': random.choice(['modified', 'added', 'removed']),
                'patch': f"@@ -1,{deletions} +1,{additions} @@"  # Simple patch format
            }
            
            files_data.append(file_data)
            
            stats['additions'] += additions
            stats['deletions'] += deletions
            stats['modifications'] += 1
        
        return files_data, stats
    
    def generate_temporal_features(self, risk_level: float) -> Dict[str, Any]:
        """
        Generate temporal features with realistic patterns
        
        Args:
            risk_level: Risk level affects timing patterns
            
        Returns:
            Dict with temporal features
        """
        # Base time
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        # Generate random timestamp
        time_diff = end_date - start_date
        random_seconds = random.randint(0, int(time_diff.total_seconds()))
        commit_time = start_date + timedelta(seconds=random_seconds)
        
        # Risky commits more likely during off-hours
        if risk_level > 0.7:
            # Late night commits (22:00 - 06:00)
            if random.random() > 0.3:
                hour = random.choice(list(range(22, 24)) + list(range(0, 7)))
                commit_time = commit_time.replace(hour=hour)
        
        weekday = commit_time.weekday()  # 0=Monday, 6=Sunday
        hour = commit_time.hour
        
        # Season encoding
        month = commit_time.month
        if month in [12, 1, 2]:
            season = 0  # Winter
        elif month in [3, 4, 5]:
            season = 1  # Spring
        elif month in [6, 7, 8]:
            season = 2  # Summer
        else:
            season = 3  # Fall
        
        return {
            'timestamp': commit_time,
            'weekday': weekday,
            'hour': hour,
            'season': season,
            'is_weekend': weekday >= 5,
            'is_business_hours': 9 <= hour <= 17,
            'unix_timestamp': int(commit_time.timestamp())
        }
    
    def generate_author_info(self, risk_level: float) -> Dict[str, Any]:
        """
        Generate author information
        
        Args:
            risk_level: Affects author selection
            
        Returns:
            Author information dict
        """
        # Select author based on risk level
        if risk_level > 0.8:
            # High risk - more likely to be junior/intern
            author_pool = [a for a in self.authors if a['experience'] < 0.5]
        elif risk_level > 0.5:
            # Medium risk - mixed experience
            author_pool = [a for a in self.authors if 0.3 <= a['experience'] <= 0.8]
        else:
            # Low risk - more likely to be senior
            author_pool = [a for a in self.authors if a['experience'] > 0.6]
        
        if not author_pool:
            author_pool = self.authors
        
        author = random.choice(author_pool)
        
        return {
            'name': author['name'],
            'experience_level': author['experience'],
            'reliability_score': author['reliability'],
            'activity_score': author['activity'],
            'commits_last_month': int(author['activity'] * 50),
            'avg_commit_size': int((1 - author['experience']) * 100 + 20)
        }
    
    def calculate_labels(self, commit_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate ground truth labels based on generated features
        
        Args:
            commit_data: Generated commit data
            
        Returns:
            Dict with task labels
        """
        # Extract features for label calculation
        risk_factors = []
        complexity_factors = []
        
        # File-based factors
        file_types = commit_data['metadata']['file_types']
        for file_type, info in self.file_types.items():
            if file_type in file_types and file_types[file_type] > 0:
                risk_factors.append(info['risk_factor'] * file_types[file_type])
                complexity_factors.append(info['complexity_base'] * file_types[file_type])
        
        # Author factors
        author_info = commit_data['metadata']['author_info']
        risk_factors.append(1 - author_info['reliability_score'])
        complexity_factors.append(1 - author_info['experience_level'])
        
        # Temporal factors
        temporal = commit_data['metadata']['temporal']
        if not temporal['is_business_hours']:
            risk_factors.append(0.3)
        if temporal['is_weekend']:
            risk_factors.append(0.2)
        
        # Change size factors
        stats = commit_data['metadata']['change_stats']
        total_changes = stats['additions'] + stats['deletions']
        if total_changes > 500:
            risk_factors.append(0.4)
            complexity_factors.append(0.5)
        elif total_changes > 200:
            risk_factors.append(0.2)
            complexity_factors.append(0.3)
        
        # File count factor
        num_files = len(commit_data['metadata']['files'])
        if num_files > 15:
            risk_factors.append(0.4)
            complexity_factors.append(0.4)
        elif num_files > 8:
            risk_factors.append(0.2)
            complexity_factors.append(0.2)
        
        # Calculate final scores
        avg_risk = np.mean(risk_factors) if risk_factors else 0.5
        avg_complexity = np.mean(complexity_factors) if complexity_factors else 0.5
        
        # Generate labels
        labels = {}
        
        # Commit Risk (Binary: 0=Low, 1=High)
        labels['commit_risk'] = 1 if avg_risk > 0.6 else 0
        
        # Complexity (3 classes: 0=Low, 1=Medium, 2=High)
        if avg_complexity > 0.7:
            labels['complexity'] = 2
        elif avg_complexity > 0.4:
            labels['complexity'] = 1
        else:
            labels['complexity'] = 0
        
        # Hotspot Files (Binary: 0=No, 1=Yes)
        # Based on high-risk file types and frequency
        hotspot_score = 0
        for file_type, count in file_types.items():
            if self.file_types[file_type]['risk_factor'] > 0.7:
                hotspot_score += count
        labels['hotspot'] = 1 if hotspot_score > 3 else 0
        
        # Urgent Review (Binary: 0=No, 1=Yes)
        # High risk + (low author reliability OR critical files OR large changes)
        urgent_factors = []
        if avg_risk > 0.7:
            urgent_factors.append(1)
        if author_info['reliability_score'] < 0.7:
            urgent_factors.append(1)
        if total_changes > 300:
            urgent_factors.append(1)
        if any(self.file_types[ft]['risk_factor'] > 0.8 for ft in file_types if file_types[ft] > 0):
            urgent_factors.append(1)
        
        labels['urgent_review'] = 1 if len(urgent_factors) >= 2 else 0
        
        return labels
    
    def generate_single_commit(self, target_risk: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate a single commit with all features and labels
        
        Args:
            target_risk: Target risk level (0-1), if None then random
            
        Returns:
            Complete commit data dict
        """
        # Determine risk level
        if target_risk is None:
            risk_level = random.random()
        else:
            risk_level = max(0.0, min(1.0, target_risk + random.gauss(0, 0.1)))
        
        # Generate complexity level (correlated with risk)
        complexity_level = risk_level + random.gauss(0, 0.2)
        complexity_level = max(0.0, min(1.0, complexity_level))
        
        # Select commit type based on risk
        if risk_level > 0.8:
            commit_type = random.choices(
                ['fix', 'feature', 'refactor', 'update', 'test'],
                weights=[0.5, 0.2, 0.1, 0.1, 0.1]
            )[0]
        elif risk_level > 0.5:
            commit_type = random.choices(
                ['fix', 'feature', 'refactor', 'update', 'test'],
                weights=[0.3, 0.3, 0.2, 0.1, 0.1]
            )[0]
        else:
            commit_type = random.choices(
                ['fix', 'feature', 'refactor', 'update', 'test'],
                weights=[0.1, 0.3, 0.3, 0.2, 0.1]
            )[0]
          # Generate components
        commit_message = self.generate_commit_message(commit_type, risk_level)
        files_data, change_stats = self.generate_file_changes(risk_level, complexity_level)
        temporal_features = self.generate_temporal_features(risk_level)
        author_info = self.generate_author_info(risk_level)
        
        # Count file types
        file_types = defaultdict(int)
        for file_data in files_data:
            filename = file_data['filename']
            for file_type, info in self.file_types.items():
                for ext in info['extensions']:
                    if filename.endswith(ext):
                        file_types[file_type] += 1
                        break
        
        # Create commit data structure
        commit_data = {
            'commit_message': commit_message,
            'metadata': {
                'files': files_data,  # Now this is list of dicts as expected by MetadataProcessor
                'change_stats': change_stats,
                'file_types': dict(file_types),
                'temporal': temporal_features,
                'author_info': author_info,
                'commit_type': commit_type,
                'risk_level': risk_level,
                'complexity_level': complexity_level
            }
        }
        
        # Calculate labels
        labels = self.calculate_labels(commit_data)
        commit_data['labels'] = labels
        
        return commit_data
    
    def generate_dataset(self, num_samples: int, 
                        risk_distribution: Optional[Dict[str, float]] = None,
                        save_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate complete dataset
        
        Args:
            num_samples: Number of samples to generate
            risk_distribution: Dict with risk level distribution
            save_path: Path to save dataset
            
        Returns:
            List of commit data dicts
        """
        if risk_distribution is None:
            risk_distribution = {
                'low': 0.5,      # 0.0 - 0.3
                'medium': 0.3,   # 0.3 - 0.7
                'high': 0.2      # 0.7 - 1.0
            }
        
        dataset = []
        
        for i in range(num_samples):
            # Select risk level based on distribution
            rand_val = random.random()
            if rand_val < risk_distribution['low']:
                target_risk = random.uniform(0.0, 0.3)
            elif rand_val < risk_distribution['low'] + risk_distribution['medium']:
                target_risk = random.uniform(0.3, 0.7)
            else:
                target_risk = random.uniform(0.7, 1.0)
            
            commit_data = self.generate_single_commit(target_risk)
            commit_data['id'] = i
            dataset.append(commit_data)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, default=str, ensure_ascii=False)
            print(f"Dataset saved to {save_path}")
        
        return dataset
    
    def generate_splits(self, dataset: List[Dict[str, Any]], 
                       split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                       stratify_by: str = 'commit_risk') -> Tuple[List, List, List]:
        """
        Split dataset into train/val/test with stratification
        
        Args:
            dataset: Complete dataset
            split_ratios: (train, val, test) ratios
            stratify_by: Label to stratify by
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        from sklearn.model_selection import train_test_split
        
        # Extract labels for stratification
        if stratify_by in dataset[0]['labels']:
            stratify_labels = [item['labels'][stratify_by] for item in dataset]
        else:
            stratify_labels = None
        
        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            dataset, 
            test_size=(1 - split_ratios[0]),
            stratify=stratify_labels,
            random_state=42
        )
        
        # Second split: val vs test
        val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        if stratify_labels:
            temp_labels = [item['labels'][stratify_by] for item in temp_data]
        else:
            temp_labels = None
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_ratio),
            stratify=temp_labels,
            random_state=42
        )
        
        print(f"Dataset splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def analyze_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze generated dataset statistics
        
        Args:
            dataset: Generated dataset
            
        Returns:
            Analysis results
        """
        analysis = {
            'total_samples': len(dataset),
            'label_distributions': {},
            'feature_statistics': {},
            'correlations': {}
        }
        
        # Label distributions
        for sample in dataset:
            for task_name, label in sample['labels'].items():
                if task_name not in analysis['label_distributions']:
                    analysis['label_distributions'][task_name] = defaultdict(int)
                analysis['label_distributions'][task_name][label] += 1
        
        # Feature statistics
        risk_levels = [sample['metadata']['risk_level'] for sample in dataset]
        complexity_levels = [sample['metadata']['complexity_level'] for sample in dataset]
        file_counts = [len(sample['metadata']['files']) for sample in dataset]
        change_sizes = [sample['metadata']['change_stats']['additions'] + 
                       sample['metadata']['change_stats']['deletions'] for sample in dataset]
        
        analysis['feature_statistics'] = {
            'risk_level': {
                'mean': np.mean(risk_levels),
                'std': np.std(risk_levels),
                'min': np.min(risk_levels),
                'max': np.max(risk_levels)
            },
            'complexity_level': {
                'mean': np.mean(complexity_levels),
                'std': np.std(complexity_levels),
                'min': np.min(complexity_levels),
                'max': np.max(complexity_levels)
            },
            'file_count': {
                'mean': np.mean(file_counts),
                'std': np.std(file_counts),
                'min': np.min(file_counts),
                'max': np.max(file_counts)
            },
            'change_size': {
                'mean': np.mean(change_sizes),
                'std': np.std(change_sizes),
                'min': np.min(change_sizes),
                'max': np.max(change_sizes)
            }
        }
        
        return analysis


def main():
    """Example usage"""
    generator = GitHubDataGenerator(seed=42)
    
    # Generate small dataset for testing
    print("Generating sample dataset...")
    dataset = generator.generate_dataset(
        num_samples=1000,
        risk_distribution={'low': 0.6, 'medium': 0.3, 'high': 0.1}
    )
    
    # Analyze dataset
    analysis = generator.analyze_dataset(dataset)
    print("\nDataset Analysis:")
    print(f"Total samples: {analysis['total_samples']}")
    print("\nLabel distributions:")
    for task, dist in analysis['label_distributions'].items():
        print(f"  {task}: {dict(dist)}")
    
    print("\nFeature statistics:")
    for feature, stats in analysis['feature_statistics'].items():
        print(f"  {feature}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    # Show sample
    print(f"\nSample commit:")
    sample = dataset[0]
    print(f"Message: {sample['commit_message']}")
    print(f"Files: {len(sample['metadata']['files'])} files")
    print(f"Changes: +{sample['metadata']['change_stats']['additions']} -{sample['metadata']['change_stats']['deletions']}")
    print(f"Labels: {sample['labels']}")


if __name__ == "__main__":
    main()
