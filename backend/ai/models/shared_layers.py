# Các lớp dùng chung hoặc private cho multi-task
import torch.nn as nn

class SharedDense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.linear(x))
