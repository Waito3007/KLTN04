# Dynamic loss weighting cho multi-task
import torch
import torch.nn as nn

def gradnorm_loss(losses, weights, grads, alpha=1.5):
    # losses: list of task losses
    # weights: nn.Parameter list
    # grads: list of grad norms
    avg_grad = torch.mean(torch.stack(grads))
    loss_gradnorm = 0
    for i, (l, w, g) in enumerate(zip(losses, weights, grads)):
        loss_gradnorm += torch.abs(g - avg_grad * (l.detach() / torch.mean(torch.stack(losses).detach())) ** alpha)
    return loss_gradnorm

class UncertaintyWeightingLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    def forward(self, losses):
        weighted = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted += precision * loss + self.log_vars[i]
        return weighted
