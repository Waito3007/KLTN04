# BiLSTM baseline cho so sánh
import torch
import torch.nn as nn

class BiLSTMBaseline(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        h, _ = self.lstm(x)
        out = h[:, -1, :]
        return self.fc(out)

def train_bilstm(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x = batch['input'].to(device)
        y = batch['label'].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def predict_bilstm(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['input'].to(device)
            y = batch['label'].to(device)
            out = model(x)
            preds.append(out.argmax(dim=1).cpu())
            labels.append(y.cpu())
    return torch.cat(preds), torch.cat(labels)
