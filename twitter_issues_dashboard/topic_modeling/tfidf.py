from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing.preprocess_tweets_rm import TextCleaner
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing.preprocess_tweets_rm import TextCleaner


class ClusterBasedTFIDF(TextCleaner):
    def __init__(self, text_col='text', cluster_col='cluster', n_topics=1, n_words=5):
        super().__init__(stop_words_remove=True)
        self.text_col = text_col
        self.cluster_col = cluster_col
        self.n_topics = n_topics
        self.n_words = n_words

    def fit_transform(self, df):
        
        # Create TF-IDF vectorizer with text cleaning
        vectorizer = TfidfVectorizer(
            preprocessor=self.clean_text,
            stop_words='english',
            max_features=None
        )

        # Compute TF-IDF scores for each cluster
        clusters = df[self.cluster_col].unique()
        topics_dict = {}
        for cluster in clusters:
            cluster_df = df[df[self.cluster_col] == cluster]
            corpus = cluster_df[self.text_col].values.tolist()
            tfidf_scores = vectorizer.fit_transform(corpus)

            # Get top words for the cluster
            feature_names = vectorizer.get_feature_names_out()
            top_indices = tfidf_scores.toarray().sum(axis=0).argsort()[-self.n_words:]
            top_words = [feature_names[i] for i in top_indices][::-1]
            topics_dict[cluster] = top_words * self.n_topics

        # Add topics to original DataFrame
        topic_col = 'topic'
        df[topic_col] = df[self.cluster_col].map(topics_dict)

        return df

