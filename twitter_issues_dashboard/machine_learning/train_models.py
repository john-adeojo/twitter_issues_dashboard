import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

class TransformerFineTuner:
    def __init__(self, train_loader, val_loader, model_name, num_labels, output_dir, logging_dir):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        
    def train(self):
        # Define model
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        torch.optim.AdamW
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,           
            num_train_epochs=3,              # total number of training epochs
            fp16=True,                       # precision choice to preserve memory on GPU
            gradient_accumulation_steps=4,   # batch size for calculating gradients  
            per_device_train_batch_size=4,   # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir=self.logging_dir,    # directory for storing logs
            logging_steps=25,
            evaluation_strategy="steps",
            load_best_model_at_end=True
        )

        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_loader.dataset,
            eval_dataset=self.val_loader.dataset,
        )

        # Train the model
        trainer.train()

        # Save the model and tokenizer
        trainer.save_model(self.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.save_pretrained(self.output_dir)