# Hierarchical Attention Network (HAN) - PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.bigru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.norm = nn.LayerNorm(hidden_dim*2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, num_word, embed_dim)
        h, _ = self.bigru(x)
        attn_weights = torch.softmax(self.attn(h), dim=1)  # (batch, num_word, 1)
        out = (h * attn_weights).sum(dim=1)  # (batch, hidden_dim*2)
        out = self.norm(out)
        out = self.dropout(out)
        return out, attn_weights.squeeze(-1)

class SentenceAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bigru = nn.GRU(hidden_dim*2, hidden_dim, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.norm = nn.LayerNorm(hidden_dim*2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, num_sent, hidden_dim*2)
        h, _ = self.bigru(x)
        attn_weights = torch.softmax(self.attn(h), dim=1)  # (batch, num_sent, 1)
        out = (h * attn_weights).sum(dim=1)  # (batch, hidden_dim*2)
        out = self.norm(out)
        out = self.dropout(out)
        return out, attn_weights.squeeze(-1)

class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes_dict):
        super().__init__()
        self.word_attn = WordAttention(embed_dim, hidden_dim)
        self.sent_attn = SentenceAttention(hidden_dim)
        # Task-specific heads
        self.heads = nn.ModuleDict({
            'purpose': nn.Linear(hidden_dim*2, num_classes_dict['purpose']),
            'suspicious': nn.Linear(hidden_dim*2, 2),
            'tech_tag': nn.Linear(hidden_dim*2, num_classes_dict['tech_tag']),
            'sentiment': nn.Linear(hidden_dim*2, 3)
        })

    def forward(self, x):
        # x: (batch, num_sent, num_word, embed_dim)
        batch, num_sent, num_word, embed_dim = x.size()
        x = x.view(-1, num_word, embed_dim)  # (batch*num_sent, num_word, embed_dim)
        word_vecs, word_attn = self.word_attn(x)  # (batch*num_sent, hidden*2)
        word_vecs = word_vecs.view(batch, num_sent, -1)
        sent_vec, sent_attn = self.sent_attn(word_vecs)  # (batch, hidden*2)
        out = {
            'purpose': self.heads['purpose'](sent_vec),
            'suspicious': self.heads['suspicious'](sent_vec),
            'tech_tag': self.heads['tech_tag'](sent_vec),
            'sentiment': self.heads['sentiment'](sent_vec)
        }
        return out, word_attn, sent_attn
