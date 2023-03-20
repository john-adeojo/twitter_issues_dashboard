import os
import sys
import pandas as pd
import torch
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(notebook_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)
    
from machine_learning.preprocess_ml import DataPipeline
from machine_learning.train_models import TransformerFineTuner
    
# load data
df_a = pd.read_csv(r"C:\Users\johna\anaconda3\envs\twitter-analytics-env\twitter_issues_dashboard\twitter_issues_dashboard\data\02_intermediate\training_data.csv")
mapping = pd.read_csv(r"C:\Users\johna\anaconda3\envs\twitter-analytics-env\twitter_issues_dashboard\twitter_issues_dashboard\data\02_intermediate\topic_mapping.csv")

df = df_a.merge(right=mapping, how='inner', right_on='topic_sortedc_str', left_on='topic_sortedc_str')[['cleaned_text', 'topic_hl']]
df['topic_hl_encoded'] = df['topic_hl'].astype("category").cat.codes


# prepare data sets
pipeline = DataPipeline(df, target_col='topic_hl_encoded', text='cleaned_text', model='cardiffnlp/twitter-roberta-base', save_data=True)
train_loader = pipeline.get_train_data()
val_loader = pipeline.get_val_data()
test_loader = pipeline.get_test_data()
torch.save(test_loader, r'C:\Users\johna\anaconda3\envs\twitter-analytics-env\twitter_issues_dashboard\twitter_issues_dashboard\data\03_feature_bank\test_dataset.pt')
torch.save(train_loader, r'C:\Users\johna\anaconda3\envs\twitter-analytics-env\twitter_issues_dashboard\twitter_issues_dashboard\data\03_feature_bank\train_dataset.pt')
torch.save(val_loader, r'C:\Users\johna\anaconda3\envs\twitter-analytics-env\twitter_issues_dashboard\twitter_issues_dashboard\data\03_feature_bank\val_dataset.pt')

# train model

transformer_fine_tuner = TransformerFineTuner(train_loader=train_loader, 
                                              val_loader=val_loader, 
                                              model_name='cardiffnlp/twitter-roberta-base', 
                                              num_labels=6, 
                                              output_dir=r"C:\Users\johna\OneDrive\Desktop\models_twitter_dash\output", 
                                              logging_dir=r"C:\Users\johna\OneDrive\Desktop\models_twitter_dash\logging")

transformer_fine_tuner.train()