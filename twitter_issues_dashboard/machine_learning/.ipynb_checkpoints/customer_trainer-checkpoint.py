import torch
from torch import nn
import transformers
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 13.19, 16.76, 15.98, 10.96, 15.98]).to(device)) # hard coded weight to be automated
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss