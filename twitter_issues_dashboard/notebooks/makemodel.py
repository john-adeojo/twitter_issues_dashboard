import os
import sys
import pandas as pd
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
pipeline = DataPipeline(df, target_col='topic_hl_encoded', model='cardiffnlp/twitter-xlm-roberta-base')
train_loader = pipeline.get_train_data(batch_size=16)
val_loader = pipeline.get_val_data(batch_size=16)
# test_loader = pipeline.get_test_data(batch_size=16)

# train model
train_loader = train_loader
val_loader = val_loader
model_name = 'cardiffnlp/twitter-roberta-base'
num_labels = 7
output_dir = r"C:\Users\johna\OneDrive\Desktop\models_twitter_dash\output"
logging_dir = r"C:\Users\johna\OneDrive\Desktop\models_twitter_dash\logging"


transformer_fine_tuner = TransformerFineTuner(train_loader, val_loader, model_name, num_labels, output_dir, logging_dir)
transformer_fine_tuner.train()