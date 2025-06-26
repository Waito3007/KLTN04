"""
Module định nghĩa kiến trúc mô hình mạng nơ-ron.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple


class TextEncoder(nn.Module):
    """Mô-đun mã hóa văn bản sử dụng LSTM hoặc Transformer."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 256, 
                 method: str = 'lstm', num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        """
        Khởi tạo text encoder.
        
        Args:
            vocab_size: Kích thước từ điển
            embedding_dim: Kích thước embedding
            hidden_dim: Kích thước hidden state
            method: 'lstm' hoặc 'transformer'
            num_layers: Số lớp LSTM hoặc Transformer
            num_heads: Số heads cho Transformer (chỉ dùng khi method='transformer')
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        
        self.method = method
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        if method == 'lstm':
            self.encoder = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.output_dim = hidden_dim * 2  # Bidirectional
        elif method == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = embedding_dim
        else:
            raise ValueError(f"Phương thức không hợp lệ: {method}. Chỉ hỗ trợ 'lstm' hoặc 'transformer'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor dạng [batch_size, seq_len]
            
        Returns:
            Tensor dạng [batch_size, output_dim]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(x))
        
        if self.method == 'lstm':
            # LSTM: [batch_size, seq_len, hidden_dim * 2]
            output, (hidden, _) = self.encoder(embedded)
            
            # Concatenate the final forward and backward hidden states
            # hidden: [num_layers * 2, batch_size, hidden_dim]
            hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            return hidden_cat
        else:  # transformer
            # Tạo mask cho padding tokens (giá trị 0)
            mask = (x == 0)
            
            # Transformer: [batch_size, seq_len, embedding_dim]
            output = self.encoder(embedded, src_key_padding_mask=mask)
            
            # Lấy embedding của token đầu tiên (non-padding token)
            # Alternatively, we could do mean pooling over all non-padding tokens
            return output[:, 0, :]


class MetadataEncoder(nn.Module):
    """Mô-đun mã hóa metadata."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        """
        Khởi tạo metadata encoder.
        
        Args:
            input_dim: Kích thước đầu vào
            hidden_dims: Danh sách kích thước các lớp ẩn
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.model = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor dạng [batch_size, input_dim]
            
        Returns:
            Tensor dạng [batch_size, output_dim]
        """
        return self.model(x)


class MultiHeadAttentionFusion(nn.Module):
    """Mô-đun fusion sử dụng Multi-head Attention."""
    
    def __init__(self, text_dim: int, metadata_dim: int, fusion_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        """
        Khởi tạo fusion module.
        
        Args:
            text_dim: Kích thước đầu ra của text encoder
            metadata_dim: Kích thước đầu ra của metadata encoder
            fusion_dim: Kích thước fusion
            num_heads: Số heads cho multi-head attention
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        
        # Các lớp projection
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.metadata_projection = nn.Linear(metadata_dim, fusion_dim)
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization và feed-forward
        self.layer_norm1 = nn.LayerNorm(fusion_dim)
        self.layer_norm2 = nn.LayerNorm(fusion_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = fusion_dim
    
    def forward(self, text_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_features: Tensor dạng [batch_size, text_dim]
            metadata_features: Tensor dạng [batch_size, metadata_dim]
            
        Returns:
            Tensor dạng [batch_size, fusion_dim]
        """
        # Project features to the same dimension
        text_proj = self.text_projection(text_features).unsqueeze(1)  # [batch_size, 1, fusion_dim]
        metadata_proj = self.metadata_projection(metadata_features).unsqueeze(1)  # [batch_size, 1, fusion_dim]
        
        # Concatenate features to create a sequence
        combined = torch.cat([text_proj, metadata_proj], dim=1)  # [batch_size, 2, fusion_dim]
        
        # Self-attention
        attn_output, _ = self.mha(combined, combined, combined)
        
        # Residual connection và layer normalization
        combined = self.layer_norm1(combined + self.dropout(attn_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(combined)
        combined = self.layer_norm2(combined + self.dropout(ff_output))
        
        # Lấy trung bình của sequence
        fusion_features = torch.mean(combined, dim=1)  # [batch_size, fusion_dim]
        
        return fusion_features


class CommitClassificationHead(nn.Module):
    """Đầu phân lớp cho một nhiệm vụ cụ thể."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Khởi tạo classification head.
        
        Args:
            input_dim: Kích thước đầu vào
            num_classes: Số lớp đầu ra
            hidden_dim: Kích thước lớp ẩn (nếu None thì không sử dụng lớp ẩn)
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        
        if hidden_dim is not None:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.model = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor dạng [batch_size, input_dim]
            
        Returns:
            Tensor dạng [batch_size, num_classes]
        """
        return self.model(x)


class CommitRegressionHead(nn.Module):
    """Đầu hồi quy cho một nhiệm vụ cụ thể."""
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Khởi tạo regression head.
        
        Args:
            input_dim: Kích thước đầu vào
            hidden_dim: Kích thước lớp ẩn (nếu None thì không sử dụng lớp ẩn)
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        
        if hidden_dim is not None:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.model = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor dạng [batch_size, input_dim]
            
        Returns:
            Tensor dạng [batch_size, 1]
        """
        return self.model(x)


class EnhancedMultimodalFusionModel(nn.Module):
    """Mô hình fusion đa phương thức cải tiến."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo mô hình.
        
        Args:
            config: Dict cấu hình mô hình
        """
        super().__init__()
        
        # Lưu cấu hình
        self.config = config
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=config['text_encoder']['vocab_size'],
            embedding_dim=config['text_encoder'].get('embedding_dim', 256),
            hidden_dim=config['text_encoder'].get('hidden_dim', 256),
            method=config['text_encoder'].get('method', 'lstm'),
            num_layers=config['text_encoder'].get('num_layers', 2),
            num_heads=config['text_encoder'].get('num_heads', 8),
            dropout=config['text_encoder'].get('dropout', 0.1)
        )
        
        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(
            input_dim=config['metadata_encoder']['input_dim'],
            hidden_dims=config['metadata_encoder'].get('hidden_dims', [128, 64]),
            dropout=config['metadata_encoder'].get('dropout', 0.1)
        )
        
        # Fusion module
        self.fusion = MultiHeadAttentionFusion(
            text_dim=self.text_encoder.output_dim,
            metadata_dim=self.metadata_encoder.output_dim,
            fusion_dim=config['fusion'].get('fusion_dim', 256),
            num_heads=config['fusion'].get('num_heads', 4),
            dropout=config['fusion'].get('dropout', 0.1)
        )
        
        # Task heads
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in config['task_heads'].items():
            if task_config.get('type', 'classification') == 'regression':
                self.task_heads[task_name] = CommitRegressionHead(
                    input_dim=self.fusion.output_dim,
                    hidden_dim=task_config.get('hidden_dim', 64),
                    dropout=task_config.get('dropout', 0.1)
                )
            else:
                self.task_heads[task_name] = CommitClassificationHead(
                    input_dim=self.fusion.output_dim,
                    num_classes=task_config['num_classes'],
                    hidden_dim=task_config.get('hidden_dim', 64),
                    dropout=task_config.get('dropout', 0.1)
                )
    
    def forward(self, text: torch.Tensor, metadata: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text: Tensor dạng [batch_size, seq_len]
            metadata: Tensor dạng [batch_size, metadata_dim]
            
        Returns:
            Dict chứa kết quả từ các task heads
        """
        # Mã hóa text và metadata
        text_features = self.text_encoder(text)
        metadata_features = self.metadata_encoder(metadata)
        
        # Fusion
        fusion_features = self.fusion(text_features, metadata_features)
        
        # Chạy qua các task heads
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(fusion_features)
        
        return outputs


def create_model_config(
    vocab_size: int,
    metadata_dim: int,
    task_heads: Dict[str, Dict[str, Any]],
    text_encoder_method: str = 'lstm',
    fusion_dim: int = 256
) -> Dict[str, Any]:
    """
    Tạo cấu hình cho mô hình.
    
    Args:
        vocab_size: Kích thước từ điển
        metadata_dim: Số đặc trưng metadata
        task_heads: Dict cấu hình cho các task heads
        text_encoder_method: 'lstm' hoặc 'transformer'
        fusion_dim: Kích thước fusion
        
    Returns:
        Dict cấu hình mô hình
    """
    config = {
        'text_encoder': {
            'vocab_size': vocab_size,
            'embedding_dim': 256,
            'hidden_dim': 256,
            'method': text_encoder_method,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1
        },
        'metadata_encoder': {
            'input_dim': metadata_dim,
            'hidden_dims': [128, 64],
            'dropout': 0.1
        },
        'fusion': {
            'fusion_dim': fusion_dim,
            'num_heads': 4,
            'dropout': 0.1
        },
        'task_heads': task_heads
    }
    
    return config


if __name__ == "__main__":
    # Ví dụ sử dụng
    task_heads = {
        'risk_prediction': {'num_classes': 3},
        'complexity_prediction': {'num_classes': 3},
        'hotspot_prediction': {'num_classes': 3},
        'urgency_prediction': {'num_classes': 3},
        'completeness_prediction': {'num_classes': 3},
        'estimated_effort': {'type': 'regression'}
    }
    
    config = create_model_config(
        vocab_size=10000,
        metadata_dim=50,
        task_heads=task_heads,
        text_encoder_method='transformer',
        fusion_dim=256
    )
    
    model = EnhancedMultimodalFusionModel(config)
    
    # Test forward pass
    batch_size = 16
    seq_len = 100
    metadata_dim = 50
    
    text = torch.randint(0, 10000, (batch_size, seq_len))
    metadata = torch.randn(batch_size, metadata_dim)
    
    outputs = model(text, metadata)
    
    for task_name, output in outputs.items():
        print(f"{task_name}: {output.shape}")
