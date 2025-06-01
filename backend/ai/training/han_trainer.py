import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import List, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CommitDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], max_words: int = 100, max_sents: int = 15):
        self.data = data
        self.max_words = max_words
        self.max_sents = max_sents
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['raw_text']
        labels = item['labels']
        
        # Convert labels to tensors
        purpose_label = self.encode_purpose(labels['purpose'])
        suspicious_label = torch.tensor(labels['suspicious'], dtype=torch.float)
        tech_tags = self.encode_tech_tags(labels['tech_tag'])
        sentiment_label = self.encode_sentiment(labels['sentiment'])
        
        # Tokenize and prepare text
        sentences = self.prepare_text(text)
        
        return {
            'text': sentences,
            'purpose': purpose_label,
            'suspicious': suspicious_label,
            'tech_tags': tech_tags,
            'sentiment': sentiment_label
        }
    
    def encode_purpose(self, purpose: str) -> torch.Tensor:
        purpose_map = {
            'Feature Implementation': 0,
            'Bug Fix': 1,
            'Documentation Update': 2,
            'Code Refactoring': 3,
            'Other': 4
        }
        return torch.tensor(purpose_map.get(purpose, 4), dtype=torch.long)
    
    def encode_tech_tags(self, tags: List[str]) -> torch.Tensor:
        all_tags = ['react', 'typescript', 'python', 'docker', 'database', 
                   'api', 'css', 'github', 'ci', 'vite', 'tailwind']
        encoded = torch.zeros(len(all_tags), dtype=torch.float)
        for tag in tags:
            if tag in all_tags:
                encoded[all_tags.index(tag)] = 1
        return encoded
    
    def encode_sentiment(self, sentiment: str) -> torch.Tensor:
        sentiment_map = {
            'positive': 0,
            'neutral': 1,
            'negative': 2
        }
        return torch.tensor(sentiment_map.get(sentiment, 1), dtype=torch.long)
    
    def prepare_text(self, text: str) -> torch.Tensor:
        # This is a placeholder - you'll need to implement proper text processing
        # using your embedding_loader.py and text_processor.py
        return torch.zeros((self.max_sents, self.max_words))

class HAN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_classes: Dict[str, int], dropout: float = 0.5):
        super(HAN, self).__init__()
        
        # Word-level attention
        self.word_gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Sentence-level attention
        self.sent_gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.sent_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layers
        self.purpose_classifier = nn.Linear(hidden_dim * 2, num_classes['purpose'])
        self.suspicious_classifier = nn.Linear(hidden_dim * 2, 1)
        self.tech_tags_classifier = nn.Linear(hidden_dim * 2, num_classes['tech_tags'])
        self.sentiment_classifier = nn.Linear(hidden_dim * 2, num_classes['sentiment'])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, documents):
        # Word level
        word_outputs, _ = self.word_gru(documents)
        word_attention = self.word_attention(word_outputs)
        word_weights = torch.softmax(word_attention, dim=1)
        word_vector = torch.sum(word_outputs * word_weights, dim=1)
        
        # Sentence level
        sent_outputs, _ = self.sent_gru(word_vector.unsqueeze(0))
        sent_attention = self.sent_attention(sent_outputs)
        sent_weights = torch.softmax(sent_attention, dim=1)
        doc_vector = torch.sum(sent_outputs * sent_weights, dim=1)
        
        # Apply dropout
        doc_vector = self.dropout(doc_vector)
        
        # Multiple outputs
        purpose_out = self.purpose_classifier(doc_vector)
        suspicious_out = torch.sigmoid(self.suspicious_classifier(doc_vector))
        tech_tags_out = torch.sigmoid(self.tech_tags_classifier(doc_vector))
        sentiment_out = self.sentiment_classifier(doc_vector)
        
        return {
            'purpose': purpose_out,
            'suspicious': suspicious_out,
            'tech_tags': tech_tags_out,
            'sentiment': sentiment_out
        }

class HANTrainer:
    def __init__(self, model: HAN, device: torch.device):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.purpose_criterion = nn.CrossEntropyLoss()
        self.suspicious_criterion = nn.BCELoss()
        self.tech_tags_criterion = nn.BCEWithLogitsLoss()
        self.sentiment_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters())
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch['text'])
            
            # Calculate losses
            purpose_loss = self.purpose_criterion(outputs['purpose'], batch['purpose'])
            suspicious_loss = self.suspicious_criterion(outputs['suspicious'].squeeze(), batch['suspicious'])
            tech_tags_loss = self.tech_tags_criterion(outputs['tech_tags'], batch['tech_tags'])
            sentiment_loss = self.sentiment_criterion(outputs['sentiment'], batch['sentiment'])
            
            # Combined loss
            loss = purpose_loss + suspicious_loss + tech_tags_loss + sentiment_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch['text'])
                
                # Calculate validation losses
                purpose_loss = self.purpose_criterion(outputs['purpose'], batch['purpose'])
                suspicious_loss = self.suspicious_criterion(outputs['suspicious'].squeeze(), batch['suspicious'])
                tech_tags_loss = self.tech_tags_criterion(outputs['tech_tags'], batch['tech_tags'])
                sentiment_loss = self.sentiment_criterion(outputs['sentiment'], batch['sentiment'])
                
                loss = purpose_loss + suspicious_loss + tech_tags_loss + sentiment_loss
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, checkpoint_dir: str):
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f'han_checkpoint_epoch_{epoch+1}.pth'
                )
                self.save_checkpoint(checkpoint_path)
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load training data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'training_data', 'han_training_samples.json')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    # Create dataset and dataloaders
    dataset = CommitDataset(training_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = HAN(
        vocab_size=10000,  # Set based on your vocabulary
        embed_dim=300,     # Set based on your embeddings
        hidden_dim=256,
        num_classes={
            'purpose': 5,
            'tech_tags': 11,
            'sentiment': 3
        }
    )
    
    # Initialize trainer
    trainer = HANTrainer(model, device)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(base_dir, 'models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        checkpoint_dir=checkpoint_dir
    )
    
    # Save final model
    final_model_path = os.path.join(base_dir, 'models', 'han_multitask.pth')
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
