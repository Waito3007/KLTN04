"""
Multi-Modal Fusion Network Architecture
Mô hình kết hợp thông tin văn bản và metadata với các cơ chế fusion tiên tiến
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention mechanism cho fusion giữa text và metadata
    """
    
    def __init__(self, text_dim: int, metadata_dim: int, hidden_dim: int = 128, num_heads: int = 4):
        super(CrossAttentionFusion, self).__init__()
        
        self.text_dim = text_dim
        self.metadata_dim = metadata_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.metadata_proj = nn.Linear(metadata_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, text_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (batch_size, text_dim)
            metadata_features: (batch_size, metadata_dim)
        Returns:
            fused_features: (batch_size, hidden_dim)
        """
        # Project to same dimension
        text_proj = self.text_proj(text_features)  # (batch_size, hidden_dim)
        metadata_proj = self.metadata_proj(metadata_features)  # (batch_size, hidden_dim)
        
        # Add sequence dimension for attention
        text_seq = text_proj.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        metadata_seq = metadata_proj.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Cross attention: text attends to metadata
        text_attended, _ = self.attention(text_seq, metadata_seq, metadata_seq)
        text_attended = text_attended.squeeze(1)  # (batch_size, hidden_dim)
        
        # Cross attention: metadata attends to text
        metadata_attended, _ = self.attention(metadata_seq, text_seq, text_seq)
        metadata_attended = metadata_attended.squeeze(1)  # (batch_size, hidden_dim)
        
        # Combine and normalize
        combined = self.norm1(text_attended + metadata_attended)
        
        # Feed forward
        output = self.ff(combined)
        output = self.norm2(combined + output)
        
        return output

class GatedFusion(nn.Module):
    """
    Gated Multimodal Units (GMU) for fusion
    """
    
    def __init__(self, text_dim: int, metadata_dim: int, hidden_dim: int = 128):
        super(GatedFusion, self).__init__()
        
        self.text_dim = text_dim
        self.metadata_dim = metadata_dim
        self.hidden_dim = hidden_dim
        
        # Project to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.metadata_proj = nn.Linear(metadata_dim, hidden_dim)
        
        # Gating mechanism
        self.gate_text = nn.Linear(text_dim + metadata_dim, hidden_dim)
        self.gate_metadata = nn.Linear(text_dim + metadata_dim, hidden_dim)
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, text_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (batch_size, text_dim)
            metadata_features: (batch_size, metadata_dim)
        Returns:
            fused_features: (batch_size, hidden_dim)
        """
        # Concatenate for gating
        combined = torch.cat([text_features, metadata_features], dim=1)
        
        # Compute gates
        text_gate = torch.sigmoid(self.gate_text(combined))
        metadata_gate = torch.sigmoid(self.gate_metadata(combined))
        
        # Project features
        text_proj = self.text_proj(text_features)
        metadata_proj = self.metadata_proj(metadata_features)
        
        # Apply gates
        gated_text = text_gate * text_proj
        gated_metadata = metadata_gate * metadata_proj
        
        # Combine and project
        fused = torch.cat([gated_text, gated_metadata], dim=1)
        output = self.output_proj(fused)
        
        return output

class TextBranch(nn.Module):
    """
    Nhánh xử lý văn bản với LSTM/GRU và Attention hoặc Transformer
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, 
                 method: str = "lstm", pretrained_dim: Optional[int] = None):
        super(TextBranch, self).__init__()
        
        self.method = method
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        if method == "lstm":
            # LSTM-based processing
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=4,
                batch_first=True
            )
            self.output_dim = hidden_dim * 2
            
        elif method in ["distilbert", "transformer"]:
            # Transformer-based processing
            if pretrained_dim is None:
                raise ValueError("pretrained_dim must be provided for transformer method")
            
            self.projection = nn.Linear(pretrained_dim, hidden_dim)
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
            self.output_dim = hidden_dim
            
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text_input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            text_input: Token IDs (batch_size, seq_len) for LSTM or embeddings (batch_size, embed_dim) for transformer
            attention_mask: Attention mask (batch_size, seq_len) - optional
        Returns:
            text_features: (batch_size, output_dim)
        """
        if self.method == "lstm":
            # LSTM processing
            embedded = self.embedding(text_input)  # (batch_size, seq_len, embed_dim)
            lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
            
            # Self-attention
            attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Global max pooling
            pooled = torch.max(attended, dim=1)[0]  # (batch_size, hidden_dim * 2)
            
        elif self.method in ["distilbert", "transformer"]:
            # Transformer processing
            if text_input.dim() == 2 and text_input.size(1) > 1:
                # Multiple embeddings case
                projected = self.projection(text_input)  # (batch_size, seq_len, hidden_dim)
                output = self.transformer(projected)  # (batch_size, seq_len, hidden_dim)
                pooled = torch.mean(output, dim=1)  # (batch_size, hidden_dim)
            else:
                # Single embedding case
                if text_input.dim() == 2:
                    projected = self.projection(text_input)  # (batch_size, hidden_dim)
                else:
                    projected = self.projection(text_input.unsqueeze(1))  # (batch_size, 1, hidden_dim)
                    projected = projected.squeeze(1)  # (batch_size, hidden_dim)
                pooled = projected
        
        return self.dropout(pooled)

class MetadataBranchV2(nn.Module):
    """
    Flexible metadata branch with configurable categorical and numerical features
    """
    
    def __init__(self, 
                 categorical_dims: Dict[str, int],
                 numerical_features: List[str],
                 embed_dim: int = 64,
                 hidden_dim: int = 128):
        super(MetadataBranchV2, self).__init__()
        
        self.categorical_dims = categorical_dims
        self.numerical_features = numerical_features
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings for categorical features
        self.categorical_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in categorical_dims.items():
            self.categorical_embeddings[feature_name] = nn.Embedding(vocab_size, embed_dim)
          # Projection for numerical features - dynamic sizing
        # We'll set this in the first forward pass when we know the actual dimension
        self.numerical_proj = None
        self.numerical_dim = None
        
        # Combine all metadata
        total_embed_dim = hidden_dim + embed_dim * len(categorical_dims)
        self.combine_layers = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, metadata_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            metadata_features: Dict containing features for categorical and numerical data
        Returns:
            metadata_vector: (batch_size, hidden_dim)
        """
        feature_list = []        # Process numerical features
        if 'numerical_features' in metadata_features:
            # Handle case where we have a single tensor with all numerical features
            numerical_tensor = metadata_features['numerical_features']
            if len(numerical_tensor.shape) == 1:
                numerical_tensor = numerical_tensor.unsqueeze(0)  # Add batch dim if missing
            
            # Initialize numerical projection layer if not already done
            if self.numerical_proj is None:
                self.numerical_dim = numerical_tensor.shape[-1]
                self.numerical_proj = nn.Linear(self.numerical_dim, self.hidden_dim).to(numerical_tensor.device)
            
            numerical_proj = self.numerical_proj(numerical_tensor)
            feature_list.append(numerical_proj)
        else:
            # Handle case where numerical features are split into individual tensors
            numerical_data = []
            for feature_name in self.numerical_features:
                if feature_name in metadata_features:
                    numerical_data.append(metadata_features[feature_name].unsqueeze(-1))
            
            if numerical_data:
                numerical_tensor = torch.cat(numerical_data, dim=1)
                
                # Initialize numerical projection layer if not already done
                if self.numerical_proj is None:
                    self.numerical_dim = numerical_tensor.shape[-1]
                    self.numerical_proj = nn.Linear(self.numerical_dim, self.hidden_dim).to(numerical_tensor.device)
                
                numerical_proj = self.numerical_proj(numerical_tensor)
                feature_list.append(numerical_proj)
        
        # Process categorical embeddings
        for feature_name, embedding_layer in self.categorical_embeddings.items():
            if feature_name in metadata_features:
                embed = embedding_layer(metadata_features[feature_name])
                feature_list.append(embed)
          # Combine all features
        if feature_list:
            combined = torch.cat(feature_list, dim=1)
        else:
            # Fallback if no features found
            batch_size = next(iter(metadata_features.values())).size(0)
            device = next(iter(metadata_features.values())).device
            combined = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Process through dense layers
        output = self.combine_layers(combined)
        
        return output

class MetadataBranch(nn.Module):
    """
    Nhánh xử lý metadata với embeddings và dense layers
    """
    
    def __init__(self, 
                 numerical_dim: int,
                 author_vocab_size: int,
                 season_vocab_size: int,
                 file_types_dim: int,
                 embed_dim: int = 64,
                 hidden_dim: int = 128):
        super(MetadataBranch, self).__init__()
        
        self.numerical_dim = numerical_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings for categorical features
        self.author_embedding = nn.Embedding(author_vocab_size, embed_dim)
        self.season_embedding = nn.Embedding(season_vocab_size, embed_dim)
        
        # Projection for numerical features
        self.numerical_proj = nn.Linear(numerical_dim, hidden_dim)
        
        # Projection for file types (multi-hot encoded)
        self.file_types_proj = nn.Linear(file_types_dim, embed_dim)
        
        # Combine all metadata
        total_embed_dim = hidden_dim + embed_dim * 3  # numerical + author + season + file_types
        self.combine_layers = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, metadata_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            metadata_features: Dict containing numerical_features, author_encoded, season_encoded, file_types_encoded
        Returns:
            metadata_vector: (batch_size, hidden_dim)
        """
        # Process numerical features
        numerical = self.numerical_proj(metadata_features['numerical_features'])
        
        # Process categorical embeddings
        author_embed = self.author_embedding(metadata_features['author_encoded'])
        season_embed = self.season_embedding(metadata_features['season_encoded'])
        file_types_embed = self.file_types_proj(metadata_features['file_types_encoded'])
        
        # Combine all features
        combined = torch.cat([numerical, author_embed, season_embed, file_types_embed], dim=1)
        
        # Process through dense layers
        output = self.combine_layers(combined)
        
        return output

class TaskSpecificHead(nn.Module):
    """
    Task-specific classification head
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super(TaskSpecificHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

class MultiModalFusionNetwork(nn.Module):
    """
    Main Multi-Modal Fusion Network
    """
    
    def __init__(self, config: Dict = None, **kwargs):
        """
        Initialize MultiModalFusionNetwork with flexible configuration
        
        Args:
            config: Configuration dictionary (new format)
            **kwargs: Backward compatibility parameters (old format)
        """
        super(MultiModalFusionNetwork, self).__init__()
        
        # Handle both new config format and old parameter format
        if config is not None:
            self.config = config
            
            # Extract configurations
            text_config = config['text_encoder']
            metadata_config = config['metadata_encoder']
            fusion_config = config['fusion']
            task_configs = config['task_heads']
            
            # Text branch
            self.text_branch = TextBranch(
                vocab_size=text_config['vocab_size'],
                embed_dim=text_config['embedding_dim'],
                hidden_dim=text_config['hidden_dim'],
                method=text_config.get('method', 'lstm'),
                pretrained_dim=text_config.get('pretrained_dim', None)
            )
            
            # Metadata branch - use flexible version
            self.metadata_branch = MetadataBranchV2(
                categorical_dims=metadata_config['categorical_dims'],
                numerical_features=metadata_config['numerical_features'],
                embed_dim=metadata_config['embedding_dim'],
                hidden_dim=metadata_config['hidden_dim']
            )
            
            # Fusion mechanism
            fusion_method = fusion_config.get('method', 'cross_attention')
            fusion_hidden_dim = fusion_config.get('fusion_dim', 128)
            
        else:
            # Backward compatibility - use old parameter format
            text_method = kwargs.get('text_method', 'lstm')
            vocab_size = kwargs.get('vocab_size', 10000)
            text_embed_dim = kwargs.get('text_embed_dim', 128)
            text_hidden_dim = kwargs.get('text_hidden_dim', 128)
            pretrained_text_dim = kwargs.get('pretrained_text_dim', None)
            
            numerical_dim = kwargs.get('numerical_dim', 34)
            author_vocab_size = kwargs.get('author_vocab_size', 1000)
            season_vocab_size = kwargs.get('season_vocab_size', 4)
            file_types_dim = kwargs.get('file_types_dim', 100)
            metadata_embed_dim = kwargs.get('metadata_embed_dim', 64)
            metadata_hidden_dim = kwargs.get('metadata_hidden_dim', 128)
            
            fusion_method = kwargs.get('fusion_method', 'cross_attention')
            fusion_hidden_dim = kwargs.get('fusion_hidden_dim', 128)
            task_configs = kwargs.get('task_configs', {})
            
            # Text branch
            self.text_branch = TextBranch(
                vocab_size=vocab_size,
                embed_dim=text_embed_dim,
                hidden_dim=text_hidden_dim,
                method=text_method,
                pretrained_dim=pretrained_text_dim
            )
            
            # Metadata branch - use old version for compatibility
            self.metadata_branch = MetadataBranch(
                numerical_dim=numerical_dim,
                author_vocab_size=author_vocab_size,
                season_vocab_size=season_vocab_size,
                file_types_dim=file_types_dim,
                embed_dim=metadata_embed_dim,
                hidden_dim=metadata_hidden_dim
            )
        
        # Common fusion setup
        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(
                text_dim=self.text_branch.output_dim,
                metadata_dim=self.metadata_branch.output_dim,
                hidden_dim=fusion_hidden_dim
            )
            fusion_output_dim = fusion_hidden_dim
        elif fusion_method == "gated":
            self.fusion = GatedFusion(
                text_dim=self.text_branch.output_dim,
                metadata_dim=self.metadata_branch.output_dim,
                hidden_dim=fusion_hidden_dim
            )
            fusion_output_dim = fusion_hidden_dim
        else:  # concat
            self.fusion = None
            fusion_output_dim = self.text_branch.output_dim + self.metadata_branch.output_dim
        
        # Task-specific heads - support both formats
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in task_configs.items():
            if isinstance(task_config, dict):
                if 'num_classes' in task_config:
                    num_classes = task_config['num_classes']
                elif 'classes' in task_config:
                    num_classes = len(task_config['classes'])
                else:
                    num_classes = 2  # default
            else:
                # Old format: direct number
                num_classes = task_config
            
            self.task_heads[task_name] = TaskSpecificHead(
                input_dim=fusion_output_dim,
                num_classes=num_classes
            )
    
    def forward(self, text_input: torch.Tensor, metadata_input: Dict[str, torch.Tensor], 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_input: Text tokens or embeddings
            metadata_input: Dict of metadata features
            attention_mask: Optional attention mask for text
            
        Returns:
            outputs: Dict mapping task names to logits
        """
        # Process text branch
        text_features = self.text_branch(text_input, attention_mask)
        
        # Process metadata branch
        metadata_features = self.metadata_branch(metadata_input)
        
        # Fusion
        if self.fusion is not None:
            fused_features = self.fusion(text_features, metadata_features)
        else:
            # Simple concatenation
            fused_features = torch.cat([text_features, metadata_features], dim=1)
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(fused_features)
        
        return outputs
    
    def get_fusion_features(self, text_input: torch.Tensor, metadata_input: Dict[str, torch.Tensor],
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get fused features for analysis/visualization
        """
        text_features = self.text_branch(text_input, attention_mask)
        metadata_features = self.metadata_branch(metadata_input)
        
        if self.fusion is not None:
            fused_features = self.fusion(text_features, metadata_features)
        else:
            fused_features = torch.cat([text_features, metadata_features], dim=1)
        
        return fused_features
