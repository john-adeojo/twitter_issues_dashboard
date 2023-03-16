import pandas as pd
import plotly.express as px
from umap.umap_ import UMAP
import hdbscan
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TopicModelingPipeline:
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def generate_umap_embeddings(self, n_components=5, n_neighbors=15, metric='cosine', min_dist=0.1):
        # Generate BERT embeddings for the cleaned text column
        embeddings = []
        for text in self.df['cleaned_text']:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)[1].detach().numpy()
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

    def visualize_clusters(self, umap_embeddings, cluster_labels):
        # Visualize the HDBSCAN clusters against the UMAP embeddings
        fig = px.scatter(
            x=umap_embeddings[:,0],
            y=umap_embeddings[:,1],
            color=cluster_labels,
            hover_data=[self.df['cleaned_text']]
        )
        fig.show()