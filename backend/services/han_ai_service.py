# backend/services/han_ai_service.py
"""
HAN AI Service - Service layer for HAN model integration
Provides high-level API for commit analysis and project management AI features
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from functools import lru_cache
import torch
import torch.nn as nn
import numpy as np
import re
from collections import Counter

logger = logging.getLogger(__name__)

# HAN Model Classes - integrated directly to avoid import issues
class SimpleHANModel(nn.Module):
    """
    Simplified Hierarchical Attention Network for commit classification
    """
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=None):
        super(SimpleHANModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Word-level LSTM
        self.word_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Word-level attention
        self.word_attention = nn.Linear(hidden_dim * 2, 1)
        
        # Sentence-level LSTM
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        
        # Sentence-level attention
        self.sentence_attention = nn.Linear(hidden_dim * 2, 1)
        
        # Multi-task classification heads
        self.classifiers = nn.ModuleDict()
        if num_classes:
            for task, num_class in num_classes.items():
                self.classifiers[task] = nn.Linear(hidden_dim * 2, num_class)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, max_sentences, max_words = input_ids.size()
        
        # Reshape for word-level processing
        input_ids = input_ids.view(-1, max_words)  # (batch_size * max_sentences, max_words)
        
        # Word embeddings
        embedded = self.embedding(input_ids)  # (batch_size * max_sentences, max_words, embed_dim)
        
        # Word-level LSTM
        word_output, _ = self.word_lstm(embedded)  # (batch_size * max_sentences, max_words, hidden_dim * 2)
        
        # Word-level attention
        word_attention_weights = torch.softmax(self.word_attention(word_output), dim=1)
        sentence_vectors = torch.sum(word_attention_weights * word_output, dim=1)  # (batch_size * max_sentences, hidden_dim * 2)
        
        # Reshape back to sentence level
        sentence_vectors = sentence_vectors.view(batch_size, max_sentences, -1)  # (batch_size, max_sentences, hidden_dim * 2)
        
        # Sentence-level LSTM
        sentence_output, _ = self.sentence_lstm(sentence_vectors)  # (batch_size, max_sentences, hidden_dim * 2)
        
        # Sentence-level attention
        sentence_attention_weights = torch.softmax(self.sentence_attention(sentence_output), dim=1)
        document_vector = torch.sum(sentence_attention_weights * sentence_output, dim=1)  # (batch_size, hidden_dim * 2)
        
        document_vector = self.dropout(document_vector)
        
        # Multi-task outputs
        outputs = {}
        for task, classifier in self.classifiers.items():
            outputs[task] = classifier(document_vector)
        
        return outputs

class SimpleTokenizer:
    """Simple tokenizer for commit messages"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
        
    def fit(self, texts):
        """Build vocabulary from texts"""
        logger.info("Building vocabulary...")
        
        for text in texts:
            # Split into sentences
            sentences = self.split_sentences(text)
            for sentence in sentences:
                words = self.tokenize_words(sentence)
                self.word_counts.update(words)
        
        # Keep most frequent words
        most_common = self.word_counts.most_common(self.vocab_size - 2)
        
        for i, (word, count) in enumerate(most_common):
            idx = i + 2  # Start from 2 (after PAD and UNK)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        logger.info(f"Vocabulary built with {len(self.word_to_idx)} words")
        
    def split_sentences(self, text):
        """Split text into sentences"""
        # Simple sentence splitting for commit messages
        sentences = re.split(r'[.!?;]|\\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [text]
    
    def tokenize_words(self, sentence):
        """Tokenize sentence into words"""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', sentence.lower())
        return words
    
    def encode_text(self, text, max_sentences, max_words):
        """Encode text to token ids"""
        sentences = self.split_sentences(text)
        
        # Pad or truncate sentences
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        while len(sentences) < max_sentences:
            sentences.append("")
        
        encoded_sentences = []
        for sentence in sentences:
            words = self.tokenize_words(sentence)
            
            # Convert words to indices
            word_ids = []
            for word in words:
                word_ids.append(self.word_to_idx.get(word, 1))  # 1 is UNK
            
            # Pad or truncate words
            if len(word_ids) > max_words:
                word_ids = word_ids[:max_words]
            while len(word_ids) < max_words:
                word_ids.append(0)  # 0 is PAD
            
            encoded_sentences.append(word_ids)
        
        return encoded_sentences

class HANAIService:
    """
    Service class for HAN-based AI analysis in project management
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HANAIService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def __init__(self):
        # __init__ will be called every time HANAIService() is called, but _initialize_model only once
        pass
    
    def _initialize_model(self):
        """Initialize HAN model with error handling"""
        try:
            success = self._load_han_model()
            if success:
                self.is_model_loaded = True
                logger.info("HAN model loaded successfully")
            else:
                logger.warning("HAN model not available - using mock responses")
                self.is_model_loaded = False
                
        except Exception as e:
            logger.error(f"Failed to load HAN model: {e}")
            self.is_model_loaded = False
    
    def _load_han_model(self):
        """Load HAN model from file"""
        try:
            # Find model path
            current_dir = Path(__file__).parent
            model_path = current_dir.parent / "ai" / "models" / "han_github_model" / "best_model.pth"
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return False
            
            logger.info(f"Loading HAN model from: {model_path}")
            
            # Register classes for unpickling
            import sys
            import types
            
            # Create a mock module to register classes
            mock_main = types.ModuleType('__main__')
            mock_main.SimpleTokenizer = SimpleTokenizer
            mock_main.SimpleHANModel = SimpleHANModel
            sys.modules['__main__'] = mock_main
            
            # Also register in torch serialization
            try:
                torch.serialization.add_safe_globals([SimpleTokenizer, SimpleHANModel])
            except:
                pass
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model components
            if 'tokenizer' not in checkpoint:
                logger.error("Tokenizer not found in checkpoint. Creating default tokenizer.")
                # Create a default tokenizer as fallback
                self.tokenizer = SimpleTokenizer()
                # Set up basic vocabulary
                self.tokenizer.word_to_idx = checkpoint.get('vocab', {'<PAD>': 0, '<UNK>': 1})
            else:
                self.tokenizer = checkpoint['tokenizer']
            
            self.label_encoders = checkpoint.get('label_encoders', {})
            model_state = checkpoint['model_state_dict']
            num_classes = checkpoint.get('num_classes', {})
            
            if not num_classes:
                # Create default classes if not found
                num_classes = {
                    'type': 8,  # feat, fix, docs, style, refactor, test, chore, other
                    'area': 5,  # frontend, backend, database, testing, general
                    'impact': 3  # low, medium, high
                }
                logger.warning("Using default num_classes")
            
            # Load model architecture
            self.model = SimpleHANModel(
                vocab_size=len(self.tokenizer.word_to_idx),
                embed_dim=100,
                hidden_dim=128,
                num_classes=num_classes
            )
            
            # Load trained weights
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("HAN model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading HAN model: {e}")
            return False
    
    def _preprocess_commit_message(self, message, max_sentences=10, max_words=50):
        """Preprocess commit message for HAN model"""
        if not self.tokenizer:
            return None
        
        try:
            tokenized_sentences = self.tokenizer.encode_text(message, max_sentences, max_words)
            return torch.tensor([tokenized_sentences], dtype=torch.long).to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing message: {e}")
            return None
    
    def _predict_with_han_model(self, commit_message):
        """Predict with HAN model"""
        try:
            # Preprocess
            input_tensor = self._preprocess_commit_message(commit_message)
            if input_tensor is None:
                return None
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Decode predictions
            predictions = {}
            
            for task, output in outputs.items():
                # Get prediction probabilities
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Decode label
                if task in self.label_encoders and self.label_encoders[task]:
                    encoder_keys = list(self.label_encoders[task].keys())
                    if predicted_idx.item() < len(encoder_keys):
                        predicted_label = encoder_keys[predicted_idx.item()]
                    else:
                        predicted_label = "unknown"
                else:
                    # Fallback labels
                    if task == 'type':
                        labels = ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'other']
                    elif task == 'area':
                        labels = ['frontend', 'backend', 'database', 'testing', 'general']
                    elif task == 'impact':
                        labels = ['low', 'medium', 'high']
                    else:
                        labels = ['unknown']
                    
                    predicted_label = labels[predicted_idx.item() % len(labels)]
                
                confidence_score = confidence.item()
                
                predictions[task] = {
                    'label': predicted_label,
                    'confidence': confidence_score
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in HAN prediction: {e}")
            return None
    
    async def analyze_commit_message(self, message: str) -> Dict[str, Any]:
        """
        Analyze a single commit message
        
        Args:
            message: Commit message text
            
        Returns:
            Analysis results including category, impact, urgency
        """
        try:
            if not self.is_model_loaded:
                return self._mock_commit_analysis(message)
            
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._predict_with_han_model, 
                message
            )
            
            if result is None:
                return self._mock_commit_analysis(message)
            
            # Convert HAN results to the application's expected format
            # The raw 'result' from the model has keys: 'commit_type', 'purpose', 'sentiment', 'tech_tag'
            commit_type_pred = result.get('commit_type', {})
            purpose_pred = result.get('purpose', {})
            tech_tag_pred = result.get('tech_tag', {})
            sentiment_pred = result.get('sentiment', {})

            # Start with the model's main prediction for commit type
            final_type = commit_type_pred.get('label', 'other')

            # If the main type is 'other', use the 'purpose' prediction to refine it.
            if final_type == 'other':
                purpose_label = purpose_pred.get('label', 'Other').lower()
                if 'fix' in purpose_label:
                    final_type = 'fix'
                elif 'feature' in purpose_label:
                    final_type = 'feat'
                elif 'doc' in purpose_label:
                    final_type = 'docs'
                elif 'test' in purpose_label:
                    final_type = 'test'
                elif 'refactor' in purpose_label:
                    final_type = 'refactor'
                elif 'style' in purpose_label:
                    final_type = 'style'
                elif 'build' in purpose_label or 'ci' in purpose_label:
                    final_type = 'chore'

            # Map sentiment to a simple impact score
            sentiment_label = sentiment_pred.get('label', 'neutral')
            impact_map = {'positive': 'low', 'neutral': 'medium', 'negative': 'high'}
            final_impact = impact_map.get(sentiment_label, 'medium')

            analysis_result = {
                'type': final_type,
                'category': final_type,  # Keep for compatibility
                'tech_area': tech_tag_pred.get('label', 'general'),
                'impact': final_impact,
                'urgency': 'normal',  # Urgency is not predicted by this model
                'confidence': {
                    'type': commit_type_pred.get('confidence', 0.0),
                    'area': tech_tag_pred.get('confidence', 0.0),
                    'impact': sentiment_pred.get('confidence', 0.0)
                },
                'raw_model_output': result # Include raw output for debugging
            }
            
            return {
                'success': True,
                'message': message,
                'analysis': analysis_result,
                'model_version': 'HAN-v1',
                'confidence': analysis_result['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing commit: {e}")
            return {
                'success': False,
                'message': message,
                'error': str(e),
                'analysis': self._mock_commit_analysis(message)['analysis']
            }
    
    async def analyze_commits_batch(self, messages: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple commit messages in batch
        
        Args:
            messages: List of commit message texts
            
        Returns:
            Batch analysis results
        """
        results = []
        
        for message in messages:
            analysis = await self.analyze_commit_message(message)
            results.append(analysis)
        
        # Generate batch statistics
        stats = self._calculate_batch_statistics(results)
        
        return {
            'success': True,
            'total_commits': len(messages),
            'results': results,
            'statistics': stats
        }
    
    # Alias for backward compatibility
    analyze_commits = analyze_commits_batch
    
    async def analyze_developer_patterns(self, developer_commits: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze commit patterns for each developer
        
        Args:
            developer_commits: Dict mapping developer name to list of commits
            
        Returns:
            Developer pattern analysis
        """
        developer_profiles = {}
        
        for developer, commits in developer_commits.items():
            if not commits:
                continue
                
            # Analyze all commits for this developer
            batch_result = await self.analyze_commits_batch(commits)
            
            # Create developer profile
            profile = self._create_developer_profile(commits, batch_result)
            developer_profiles[developer] = profile
        
        return {
            'success': True,
            'developer_profiles': developer_profiles,
            'total_developers': len(developer_profiles)
        }
    
    async def suggest_task_assignment(self, tasks: List[Dict], developers: List[Dict]) -> Dict[str, Any]:
        """
        Suggest task assignments based on commit analysis and developer profiles
        
        Args:
            tasks: List of task dictionaries
            developers: List of developer dictionaries with commit history
            
        Returns:
            Task assignment suggestions
        """
        try:
            # Analyze developer patterns first
            developer_commits = {}
            for dev in developers:
                developer_commits[dev['login']] = dev.get('recent_commits', [])
            
            developer_analysis = await self.analyze_developer_patterns(developer_commits)
            
            # Generate task assignments
            assignments = []
            for task in tasks:
                assignment = self._match_task_to_developer(
                    task, 
                    developer_analysis['developer_profiles']
                )
                assignments.append(assignment)
            
            return {
                'success': True,
                'assignments': assignments,
                'developer_analysis': developer_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in task assignment: {e}")
            return {
                'success': False,
                'error': str(e),
                'assignments': self._mock_task_assignments(tasks, developers)
            }
    
    async def generate_project_insights(self, project_data: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive project insights based on commit analysis
        
        Args:
            project_data: Project data including commits, contributors, etc.
            
        Returns:
            Project insights and recommendations
        """
        try:
            all_commits = project_data.get('commits', [])
            contributors = project_data.get('contributors', [])
            
            # Analyze all commits
            commit_messages = [commit.get('message', '') for commit in all_commits]
            batch_analysis = await self.analyze_commits_batch(commit_messages)
            
            # Generate insights
            insights = {
                'commit_analysis': batch_analysis,
                'code_quality_trends': self._analyze_quality_trends(batch_analysis),
                'team_collaboration': self._analyze_team_collaboration(all_commits, contributors),
                'project_health': self._assess_project_health(batch_analysis),
                'recommendations': self._generate_recommendations(batch_analysis)
            }
            
            return {
                'success': True,
                'project_name': project_data.get('name', 'Unknown'),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error generating project insights: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _mock_commit_analysis(self, message: str) -> Dict[str, Any]:
        """Mock analysis when model is not available"""
        # Simple keyword-based mock analysis
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['fix', 'bug', 'error', 'issue']):
            category = 'bug'
            impact = 'medium'
            urgency = 'high'
        elif any(word in message_lower for word in ['feat', 'feature', 'add', 'new']):
            category = 'feature'
            impact = 'high'
            urgency = 'medium'
        elif any(word in message_lower for word in ['docs', 'doc', 'readme']):
            category = 'docs'
            impact = 'low'
            urgency = 'low'
        elif any(word in message_lower for word in ['test', 'spec']):
            category = 'test'
            impact = 'medium'
            urgency = 'low'
        else:
            category = 'chore'
            impact = 'low'
            urgency = 'medium'
        
        return {
            'success': True,
            'analysis': {
                'category': category,
                'impact': impact,
                'urgency': urgency,
                'is_mock': True
            }
        }
    
    def _calculate_batch_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics from batch analysis results"""
        if not results:
            return {}
        
        categories = {}
        impacts = {}
        urgencies = {}
        successful = 0
        
        for result in results:
            if result.get('success', False):
                successful += 1
                analysis = result.get('analysis', {})
                
                # Count categories
                category = analysis.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                # Count impacts
                impact = analysis.get('impact', 'unknown')
                impacts[impact] = impacts.get(impact, 0) + 1
                
                # Count urgencies
                urgency = analysis.get('urgency', 'unknown')
                urgencies[urgency] = urgencies.get(urgency, 0) + 1
        
        return {
            'total_analyzed': len(results),
            'successful_analyses': successful,
            'success_rate': successful / len(results) if results else 0,
            'category_distribution': categories,
            'impact_distribution': impacts,
            'urgency_distribution': urgencies
        }
    
    def _create_developer_profile(self, commits: List[str], analysis_result: Dict) -> Dict[str, Any]:
        """Create developer profile from commit analysis"""
        stats = analysis_result.get('statistics', {})
        
        return {
            'total_commits': len(commits),
            'preferred_categories': self._get_top_categories(stats.get('category_distribution', {})),
            'impact_pattern': self._get_top_categories(stats.get('impact_distribution', {})),
            'urgency_pattern': self._get_top_categories(stats.get('urgency_distribution', {})),
            'activity_score': min(len(commits) / 10, 10),  # Scale 0-10
            'specialization': self._determine_specialization(stats)
        }
    
    def _get_top_categories(self, distribution: Dict[str, int], top_n: int = 3) -> List[str]:
        """Get top N categories from distribution"""
        if not distribution:
            return []
        
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:top_n]]
    
    def _determine_specialization(self, stats: Dict) -> str:
        """Determine developer specialization based on commit patterns"""
        categories = stats.get('category_distribution', {})
        
        if not categories:
            return 'generalist'
        
        top_category = max(categories.items(), key=lambda x: x[1])
        total_commits = sum(categories.values())
        
        if top_category[1] / total_commits > 0.5:
            return f"{top_category[0]}_specialist"
        else:
            return 'generalist'
    
    def _match_task_to_developer(self, task: Dict, developer_profiles: Dict) -> Dict[str, Any]:
        """Match a task to the best developer based on profiles"""
        task_type = task.get('type', 'feature').lower()
        task_priority = task.get('priority', 'medium').lower()
        
        best_match = None
        best_score = 0
        
        for dev_name, profile in developer_profiles.items():
            score = self._calculate_match_score(task_type, task_priority, profile)
            if score > best_score:
                best_score = score
                best_match = dev_name
        
        return {
            'task_title': task.get('title', 'Untitled'),
            'task_type': task_type,
            'recommended_developer': best_match or 'No suitable match',
            'confidence_score': best_score,
            'reasoning': self._generate_assignment_reasoning(task, best_match, developer_profiles.get(best_match, {}))
        }
    
    def _calculate_match_score(self, task_type: str, task_priority: str, profile: Dict) -> float:
        """Calculate match score between task and developer"""
        score = 0.0
        
        # Check category preference
        preferred_categories = profile.get('preferred_categories', [])
        if task_type in preferred_categories:
            score += 0.4
        
        # Check specialization
        specialization = profile.get('specialization', '')
        if task_type in specialization:
            score += 0.3
        
        # Activity score
        activity_score = profile.get('activity_score', 0)
        score += (activity_score / 10) * 0.3
        
        return score
    
    def _generate_assignment_reasoning(self, task: Dict, developer: str, profile: Dict) -> str:
        """Generate reasoning for task assignment"""
        if not developer or not profile:
            return "No suitable developer found based on commit analysis"
        
        specialization = profile.get('specialization', 'generalist')
        activity_score = profile.get('activity_score', 0)
        
        return f"Recommended {developer} based on {specialization} specialization and activity score of {activity_score:.1f}/10"
    
    def _mock_task_assignments(self, tasks: List[Dict], developers: List[Dict]) -> List[Dict]:
        """Generate mock task assignments when model is unavailable"""
        assignments = []
        
        for i, task in enumerate(tasks):
            dev_index = i % len(developers) if developers else 0
            developer = developers[dev_index]['login'] if developers else 'Unknown'
            
            assignments.append({
                'task_title': task.get('title', 'Untitled'),
                'recommended_developer': developer,
                'confidence_score': 0.5,
                'reasoning': 'Mock assignment - HAN model not available'
            })
        
        return assignments
    
    def _analyze_quality_trends(self, analysis: Dict) -> Dict[str, Any]:
        """Analyze code quality trends from commit analysis"""
        stats = analysis.get('statistics', {})
        categories = stats.get('category_distribution', {})
        
        bug_ratio = categories.get('bug', 0) / max(stats.get('total_analyzed', 1), 1)
        test_ratio = categories.get('test', 0) / max(stats.get('total_analyzed', 1), 1)
        
        quality_score = max(0, 1 - bug_ratio + test_ratio * 0.5)
        
        return {
            'quality_score': round(quality_score, 2),
            'bug_fix_ratio': round(bug_ratio, 2),
            'test_coverage_indicator': round(test_ratio, 2),
            'trend': 'improving' if quality_score > 0.7 else 'needs_attention'
        }
    
    def _analyze_team_collaboration(self, commits: List[Dict], contributors: List[Dict]) -> Dict[str, Any]:
        """Analyze team collaboration patterns"""
        return {
            'total_contributors': len(contributors),
            'commit_distribution': 'balanced',  # Simplified
            'collaboration_score': 0.8  # Mock score
        }
    
    def _assess_project_health(self, analysis: Dict) -> Dict[str, Any]:
        """Assess overall project health"""
        stats = analysis.get('statistics', {})
        success_rate = stats.get('success_rate', 0)
        
        health_score = success_rate * 0.8 + 0.2  # Base score
        
        return {
            'health_score': round(health_score, 2),
            'status': 'healthy' if health_score > 0.7 else 'needs_attention',
            'analysis_coverage': f"{stats.get('successful_analyses', 0)}/{stats.get('total_analyzed', 0)}"
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate project recommendations based on analysis"""
        recommendations = []
        stats = analysis.get('statistics', {})
        categories = stats.get('category_distribution', {})
        
        total = sum(categories.values()) if categories else 1
        
        if categories.get('bug', 0) / total > 0.3:
            recommendations.append("Consider increasing code review practices to reduce bug fixes")
        
        if categories.get('test', 0) / total < 0.1:
            recommendations.append("Increase test coverage to improve code quality")
        
        if categories.get('docs', 0) / total < 0.05:
            recommendations.append("Improve documentation practices")
        
        if not recommendations:
            recommendations.append("Project shows good development practices")
        
        return recommendations

# Singleton instance
_han_ai_service = None

def get_han_ai_service() -> HANAIService:
    """Get singleton instance of HAN AI Service"""
    global _han_ai_service
    if _han_ai_service is None:
        _han_ai_service = HANAIService()
    return _han_ai_service
