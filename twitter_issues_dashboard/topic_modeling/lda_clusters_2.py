from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from data_processing.preprocess_tweets import TextCleaner
from nltk.corpus import stopwords


class ClusterBasedLDA(TextCleaner):
    def __init__(self, num_topics=5, passes=20, text='text', cluster='cluster'):
        super().__init__(stop_words_remove=False)
        self.num_topics = num_topics
        self.passes = passes
        self.cluster = cluster
        self.text = text
        self.stop_words = set(stopwords.words('english'))

    def get_lda_topics(self, df):
        topics_dict = {}
        for cluster in df[self.cluster].unique():
            cluster_df = df[df[self.cluster] == cluster]
            texts = cluster_df[self.text].values.tolist()
            # preprocess texts here
            preprocessed_texts = [self.clean_text(text) for text in texts]
            vectorizer = CountVectorizer(stop_words='english')
            dtm = vectorizer.fit_transform(preprocessed_texts)
            lda_model = LatentDirichletAllocation(n_components=self.num_topics, max_iter=self.passes, random_state=42)
            lda_model.fit(dtm)
            feature_names = vectorizer.get_feature_names_out()
            cluster_topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                topic_words = [feature_names[i] for i in topic.argsort()[:-self.num_topics-1:-1]]
                cluster_topics.append(topic_words)
            topics_dict[cluster] = cluster_topics
        return topics_dict

    def add_lda_topics_to_df(self, df, topics_dict):
        new_df = df.copy()
        new_df['topics'] = new_df[self.cluster].apply(lambda x: topics_dict[x])
        return new_df
