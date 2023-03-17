import pandas as pd
import plotly.express as px
from umap.umap_ import UMAP
import hdbscan
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

class TopicModelingPipeline:
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
              
    def generate_umap_embeddings(self, batch_size=32, n_components=5, n_neighbors=15, metric='cosine', min_dist=0.1):
        # Generate BERT embeddings for the cleaned text column
        embeddings = []
        for i in range(0, len(self.df), batch_size):
            batch = self.df['cleaned_text'][i:i+batch_size]
            inputs = self.tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to the same device as the model
            with torch.no_grad():
                outputs = self.model(**inputs)[1].cpu().numpy()
            embeddings.append(outputs)

        # Convert embeddings to 2D array
        embeddings = np.concatenate(embeddings, axis=0)

        # Perform dimensionality reduction with UMAP
        umap_embeddings = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
            min_dist=min_dist
        ).fit_transform(embeddings)

        # Visualize the UMAP embeddings
        fig = px.scatter(x=umap_embeddings[:,0], y=umap_embeddings[:,1])
        fig.show()

        return umap_embeddings

    def generate_hdbscan_clusters(self, umap_embeddings, min_cluster_size=10, min_samples=1, cluster_selection_epsilon=0.5):
        # Generate topic clusters with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon
        ).fit(umap_embeddings)

        return clusterer.labels_

    # def visualize_clusters(self, umap_embeddings, cluster_labels):
    #     # Visualize the HDBSCAN clusters against the UMAP embeddings
    #     fig = px.scatter(
    #         x=umap_embeddings[:,0],
    #         y=umap_embeddings[:,1],
    #         color=cluster_labels,
    #         hover_data=[self.df['cleaned_text']]
    #     )
    #     fig.show()
    
    def visualize_clusters(self, umap_embeddings, cluster_labels):
        # Generate a unique color for each cluster
        unique_labels = np.unique(cluster_labels)
        num_clusters = len(unique_labels)
        colors = px.colors.qualitative.Dark24[:num_clusters]

        # Create a dictionary that maps cluster IDs to their corresponding label names
        cluster_names = {}
        for label in unique_labels:
            cluster_names[label] = f'Cluster {label}'

        # Map each label to its corresponding color and name
        colors = [colors[np.where(unique_labels == label)[0][0]] for label in cluster_labels]
        legend_labels = [cluster_names[label] for label in cluster_labels]

        # Create the scatter plot
        fig = px.scatter(
            x=umap_embeddings[:,0],
            y=umap_embeddings[:,1],
            color=legend_labels,
            hover_data=[self.df['cleaned_text']],
            color_discrete_map=cluster_names,
            template='plotly_white'
        )

        # Update the legend labels
        if len(fig.data) == num_clusters:
            for i, label in enumerate(cluster_names.values()):
                fig.data[i].name = label

        fig.show()
