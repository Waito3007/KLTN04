import torch
from tqdm import tqdm

# Huấn luyện đa nhiệm cho HAN
class MultiTaskTrainer:
    def __init__(self, model, optimizer, loss_fns, device, loss_weights=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fns = loss_fns  # dict: {task: loss_fn}
        self.device = device
        self.loss_weights = loss_weights or {k: 1.0 for k in loss_fns}

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            x = batch['input'].to(self.device)
            y = {k: v.to(self.device) for k, v in batch['labels'].items()}
            self.optimizer.zero_grad()
            out, _, _ = self.model(x)
            loss = 0
            for task, loss_fn in self.loss_fns.items():
                task_loss = loss_fn(out[task], y[task])
                loss += self.loss_weights[task] * task_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = {k: [] for k in self.loss_fns}
        all_labels = {k: [] for k in self.loss_fns}
        with torch.no_grad():
            for batch in tqdm(dataloader):
                x = batch['input'].to(self.device)
                y = {k: v.to(self.device) for k, v in batch['labels'].items()}
                out, _, _ = self.model(x)
                loss = 0
                for task, loss_fn in self.loss_fns.items():
                    task_loss = loss_fn(out[task], y[task])
                    loss += self.loss_weights[task] * task_loss
                    all_preds[task].append(out[task].cpu())
                    all_labels[task].append(y[task].cpu())
                total_loss += loss.item()
        # Gộp lại
        for task in all_preds:
            all_preds[task] = torch.cat(all_preds[task])
            all_labels[task] = torch.cat(all_labels[task])
        return total_loss / len(dataloader), all_preds, all_labels
