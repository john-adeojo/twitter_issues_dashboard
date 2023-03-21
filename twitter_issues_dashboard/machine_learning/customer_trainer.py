import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import Trainer


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, reduction='none')

        # Ensure alpha has the correct shape
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha[target].view(-1, 1)
        else:
            alpha = self.alpha

        loss = alpha * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fct = FocalLoss(alpha=torch.tensor([1.0, 13.19, 16.76, 15.98, 10.96, 15.98]).to(device), gamma=3).to(device)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 13.19, 16.76, 15.98, 10.96, 15.98]).to(device)) # hard coded weight to be automated
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss