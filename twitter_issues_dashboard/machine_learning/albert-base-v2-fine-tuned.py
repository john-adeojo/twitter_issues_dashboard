import sys 
import pandas as pd
import numpy as np
import re

df = pd.read_csv(r"C:\Users\johna\anaconda3\envs\twitter-env-2\Data\01 Raw Data\twitter_data.csv")


class DataProcessor:
    
    def __init__(self, df):
        self.df = df.copy()
    
    def drop_unknown_users(self):
        self.df = self.df.dropna(subset=["gender"])
        self.df = self.df[self.df["gender"] != "unknown"]
    
    def replace_blank_user_description(self):
        self.df["user_description"] = self.df["user_description"].fillna("no user description")
        
    def clean_text(self):
        # remove URLs
        self.df[['user_description', 'tweet_text']] = self.df[['user_description', 'tweet_text']].applymap(lambda x: re.sub(r'http\S+', '', str(x)))

        # remove corrupted characters
        self.df[['user_description', 'tweet_text']] = self.df[['user_description', 'tweet_text']].applymap(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        
        # reomvespecial characters
        special_chars = r'[@#_\|/\(\)\{\}\[\]\*]'
        self.df['user_description'] = self.df['user_description'].apply(lambda x: re.sub(special_chars, '', x))
        self.df['tweet_text'] = self.df['tweet_text'].apply(lambda x: re.sub(special_chars, '', x))
        
    def isolate_duplicates(self):
        duplicate_names = self.df[self.df.duplicated("name")]["name"]
        self.duplicates = self.df[self.df["name"].isin(duplicate_names)]
        self.df = self.df[~self.df["name"].isin(duplicate_names)]
    
    def split_data(self):
        # Split into train, validation, and test sets
        train_pct = 0.7
        val_pct = 0.15
        test_pct = 0.15

        # Set random seed for reproducibility
        np.random.seed(42)

        # Shuffle the data
        shuffled_df = self.df.sample(frac=1)

        # Determine the number of rows for each set
        num_rows = len(shuffled_df)
        train_rows = int(num_rows * train_pct)
        val_rows = int(num_rows * val_pct)
        test_rows = num_rows - train_rows - val_rows

        # Split the data into sets
        self.train_df = shuffled_df.iloc[:train_rows]
        self.val_df = shuffled_df.iloc[train_rows:train_rows+val_rows]
        self.test_df = shuffled_df.iloc[train_rows+val_rows:]
        
        # Add duplicates to training set
        self.train_df = pd.concat([self.train_df, self.duplicates])
        
        # Ensure no overlap in names
        train_names = set(self.train_df["name"])
        val_names = set(self.val_df["name"])
        test_names = set(self.test_df["name"])
        
        assert len(train_names.intersection(val_names)) == 0, "Overlap in names between train and validation sets"
        assert len(train_names.intersection(test_names)) == 0, "Overlap in names between train and test sets"
        assert len(val_names.intersection(test_names)) == 0, "Overlap in names between validation and test sets"
        
    def expand_data_set (self, df):
        # Expands the data sets by concat tweets and description into general tweet_text
        _tweets = df[["gender", "tweet_text"]].rename(columns={"tweet_text": "general_twitter_text"})
        _descriptions = df[["gender", "user_description"]].rename(columns={"user_description": "general_twitter_text"})
        _full = pd.concat([_tweets, _descriptions])
        return _full
        
    def process_data(self):
        
        self.drop_unknown_users()
        self.replace_blank_user_description()
        self.clean_text()
        self.isolate_duplicates()
        self.split_data()
        
# Pre-process data
processor = DataProcessor(df)
processor.process_data()

# Create train, validation & test
train_df = processor.train_df
validation_df = processor.val_df
test_df = processor.test_df

# Expand train, validation & test 
expanded_train_df = processor.expand_data_set(train_df)
expanded_validation_df = processor.expand_data_set(validation_df)
expanded_test_df = processor.expand_data_set(test_df)


# Create labels and features
def features_labels(df):
    y = list(df["gender"].astype("category").cat.codes)
    X = list((df["general_twitter_text"]))
    return y, X

train_labels, train_text = features_labels(df=expanded_train_df)
validation_labels, validation_text = features_labels(df=expanded_validation_df)
test_labels, test_text = features_labels(df=expanded_test_df)


# Pytorch Dataset class
import torch
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
# Model fine tunining: disitlibert-base-multilingual-cased    
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification


# set model and tokenizer
MODEL ="albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
# Create pytorch Dataset
train_encodings = tokenizer(train_text, truncation=True, padding=True)
val_encodings = tokenizer(validation_text, truncation=True, padding=True)
test_encodings = tokenizer(test_text, truncation=True, padding=True)

train_dataset = TwitterDataset(train_encodings, train_labels)
validation_dataset = TwitterDataset(val_encodings, validation_labels)
test_dataset = TwitterDataset(test_encodings, test_labels)

# Fine tune model
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)

output_dir = r"C:\Users\johna\anaconda3\envs\twitter-env-2\models\albert-base-v2\results"
logging_dir = r"C:\Users\johna\anaconda3\envs\twitter-env-2\models\albert-base-v2\logs"

training_args = TrainingArguments(
    output_dir=output_dir,           
    num_train_epochs=6,              # total number of training epochs
    fp16=True,                       # precision choice to preserve memory on GPU
    gradient_accumulation_steps=4,   # batch size for calculating gradients  
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=logging_dir,         # directory for storing logs
    logging_steps=25,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    
)


trainer = Trainer(
    model=model,                         # the instantiated  Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=validation_dataset      # evaluation dataset                              
    
)

trainer.train()


trainer.save_model(r"C:\Users\johna\anaconda3\envs\twitter-env-2\models\albert-base-v2")
tokenizer.save_pretrained(r"C:\Users\johna\anaconda3\envs\twitter-env-2\models\albert-base-v2")