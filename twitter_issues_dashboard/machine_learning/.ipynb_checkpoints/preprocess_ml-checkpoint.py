import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class DataPipeline:
    def __init__(self, df, target_col, text, model, save_data, random_state=42):
        self.df = df
        self.model = model
        self.target_col = target_col
        self.text = text
        self.random_state = random_state
        self.save_data = save_data

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
        
        # save data sets 
        
        if self.save_data == True:
            self.train_df.to_csv(r"C:\Users\johna\anaconda3\envs\twitter-analytics-env\twitter_issues_dashboard\twitter_issues_dashboard\data\03_feature_bank\train_df.csv")
            self.val_df.to_csv(r"C:\Users\johna\anaconda3\envs\twitter-analytics-env\twitter_issues_dashboard\twitter_issues_dashboard\data\03_feature_bank\val_df.csv")
            self.test_df.to_csv(r"C:\Users\johna\anaconda3\envs\twitter-analytics-env\twitter_issues_dashboard\twitter_issues_dashboard\data\03_feature_bank\test_df.csv")

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        
    def get_encodings(self, df):
        text = list(df[self.text])
        encodings = self.tokenizer(text, truncation=True, padding=True)
        labels = list(df[self.target_col])
        return encodings, labels
        

    def get_train_data(self):
        
        encodings, labels = self.get_encodings(self.train_df)
        self.train_dataset = TextClassificationDataset(encodings, labels)
        return self.train_dataset

    def get_val_data(self):
        
        encodings, labels = self.get_encodings(self.val_df)
        self.val_dataset = TextClassificationDataset(encodings, labels)
       
        return self.val_dataset

    def get_test_data(self):
        
        encodings, labels = self.get_encodings(self.test_df)
        self.test_dataset = TextClassificationDataset(encodings, labels)
       
        return self.test_dataset


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)   
    

    
    
    
    
    
    
    
