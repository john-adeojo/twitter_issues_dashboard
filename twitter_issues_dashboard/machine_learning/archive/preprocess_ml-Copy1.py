import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class DataPipeline:
    def __init__(self, df, text, target_col, model, random_state=42):
        self.df = df
        self.model = model
        self.target_col = target_col
        self.random_state = random_state
        self.text = text

        # split into train and test sets
        self.train_df, self.test_df = train_test_split(
            self.df,
            stratify=self.df[self.target_col],
            test_size=0.2,
            random_state=self.random_state
        )

        # split train into train and validation sets
        self.train_df, self.val_df = train_test_split(
            self.train_df,
            stratify=self.train_df[self.target_col],
            test_size=0.2,
            random_state=self.random_state
        )

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def get_train_data(self):
        
        tokenized_text = self.tokenizer(list(self.train_df[self.text]), truncation=True, padding=True)
        y = list(self.train_df[self.target_col])
        
        traindataset = TextClassificationDataset(tokenized_text, y)
        
        
#         train_dataset = TextClassificationDataset(
#             self.train_df,
#             target_col=self.target_col,
#             tokenizer=self.tokenizer,
#             max_len=514,
#             model=self.model
#         )
#         train_loader = DataLoader(train_dataset, batch_size=batch_size)
        return traindataset

    def get_val_data(self):
        
        tokenized_text = self.tokenizer(list(self.val_df[self.text]), truncation=True, padding=True)
        y = list(self.val_df[self.target_col])
        
        valdataset = TextClassificationDataset(tokenized_text, y)
        
        
        
        
        # val_dataset = TextClassificationDataset(
        #     self.val_df,
        #     target_col=self.target_col,
        #     tokenizer=self.tokenizer,
        #     max_len=514,
        #     model=self.model
        # )
        # val_loader = DataLoader(val_dataset, batch_size=batch_size)
        return valdataset

    def get_test_data(self):
        
        tokenized_text = self.tokenizer(list(self.test_df[self.text]), truncation=True, padding=True)
        y = list(self.test_df[self.target_col])
        
        testdataset = TextClassificationDataset(tokenized_text, y)
        
        
        
        # test_dataset = TextClassificationDataset(
        #     self.test_df,
        #     target_col=self.target_col,
        #     tokenizer=self.tokenizer,
        #     max_len=514,
        #     model=self.model
        # )
        # test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return testdataset


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)    
    
    
    
# class TextClassificationDataset(Dataset):
#     def __init__(self, df, target_col, tokenizer, max_len, model):
#         self.df = df
#         self.target_col = target_col
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.model = model

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         text = self.df.iloc[idx]['cleaned_text']
#         target = self.df.iloc[idx][self.target_col]

#         encoding = self.tokenizer(
#             text,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )

#         input_ids = encoding['input_ids'].squeeze(0)
#         attention_mask = encoding['attention_mask'].squeeze(0)

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'target': torch.tensor(target)
#         }
