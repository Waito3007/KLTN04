import os
import torch
import json
import sys

# Đảm bảo import đúng model và processor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from multimodal_fusion.data_preprocessing.minimal_enhanced_text_processor import MinimalEnhancedTextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor

class MultiModalFusionInference:
    def __init__(self, model_dir=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or os.path.join(current_dir, 'trained_models', 'enhanced_multimodal_fusion_100k')
        self.model_path = os.path.join(self.model_dir, 'best_enhanced_model.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.text_processor = None
        self.metadata_processor = None
        self.model_config = None
        self._load_all()

    def _load_all(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_config = checkpoint['model_config']
        self.model = MultiModalFusionNetwork(config=self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        # Load text processor
        self.text_processor = MinimalEnhancedTextProcessor(
            method="lstm",
            vocab_size=self.model_config['text_encoder']['vocab_size'],
            max_length=self.model_config['text_encoder'].get('max_length', 128),
            enable_sentiment=True,
            enable_advanced_cleaning=True
        )
        self.text_processor.vocab = checkpoint.get('text_processor_vocab', {})
        # Load metadata processor
        self.metadata_processor = MetadataProcessor()
        # (Nếu cần, fit lại metadata_processor bằng metadata_samples nếu lưu trong checkpoint)

    def predict_commit_analysis(self, message, metadata=None):
        # Encode text
        text_encoded = self.text_processor.encode_text_lstm(message)
        enhanced_features = self.text_processor.extract_enhanced_features(message)
        feature_keys = [
            'length', 'word_count', 'char_count', 'digit_count', 'upper_count', 'punctuation_count',
            'has_commit_type', 'has_bug_keywords', 'has_feature_keywords', 'has_doc_keywords',
            'has_technical_keywords', 'has_ui_keywords', 'has_testing_keywords',
            'avg_word_length', 'max_word_length', 'unique_word_ratio',
            'sentiment_polarity', 'sentiment_subjectivity'
        ]
        feature_values = [float(enhanced_features.get(k, 0)) for k in feature_keys]
        enhanced_text_features = torch.tensor(feature_values, dtype=torch.float32)
        # Metadata features (dummy if not provided)
        if metadata is None:
            metadata = {}
        files_changed_count = len(metadata.get('files_mentioned', [])) if isinstance(metadata.get('files_mentioned', []), list) else 1
        metadata_features = torch.tensor([
            float(files_changed_count),
            float(metadata.get('insertions', 0)),
            float(metadata.get('deletions', 0)),
            float(metadata.get('hour_of_day', 12) / 24.0),
            float(metadata.get('day_of_week', 1) / 7.0),
            float(metadata.get('is_merge', False)),
            1.0 if metadata.get('commit_size') == 'small' else 0.0,
            1.0 if metadata.get('commit_size') == 'medium' else 0.0,
            1.0 if metadata.get('commit_size') == 'large' else 0.0,
            hash(metadata.get('author', 'unknown')) % 1000 / 1000.0
        ], dtype=torch.float32)
        # Prepare batch
        text_encoded = text_encoded.unsqueeze(0).to(self.device)
        enhanced_text_features = enhanced_text_features.unsqueeze(0).to(self.device)
        metadata_features = metadata_features.unsqueeze(0).to(self.device)
        combined_features = torch.cat([metadata_features, enhanced_text_features], dim=1)
        metadata_input = {
            'numerical_features': combined_features,
            'author': torch.zeros(1, dtype=torch.long).to(self.device)
        }
        with torch.no_grad():
            outputs = self.model(text_encoded, metadata_input)
        # Mapping output
        task_names = ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']
        label_map = {
            'risk_prediction': ['low', 'medium', 'high'],
            'complexity_prediction': ['simple', 'moderate', 'complex'],
            'hotspot_prediction': ['low', 'medium', 'high'],
            'urgency_prediction': ['low', 'medium', 'high']
        }
        result = {}
        for i, task in enumerate(task_names):
            if task in outputs:
                logits = outputs[task][0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred = int(probs.argmax())
                result[task] = label_map[task][pred]
                result[f'{task}_probs'] = {k: float(v) for k, v in zip(label_map[task], probs)}
        result['input'] = message
        return result
