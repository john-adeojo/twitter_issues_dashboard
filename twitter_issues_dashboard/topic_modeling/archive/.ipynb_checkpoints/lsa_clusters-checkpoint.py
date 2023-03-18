from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from data_processing.preprocess_tweets_rm import TextCleaner
from nltk.corpus import stopwords


class ClusterBasedLSA(TextCleaner):
    def __init__(self, num_topics, n_words_per_topic, text, cluster):
        super().__init__(stop_words_remove=False)
        self.num_topics = num_topics
        self.n_words_per_topic = n_words_per_topic
        self.cluster = cluster
        self.text = text
        self.stop_words = set(stopwords.words('english'))

    def get_lsa_topics(self, df):
        topics_dict = {}
        for cluster in df[self.cluster].unique():
            cluster_df = df[df[self.cluster] == cluster]
            texts = cluster_df[self.text].values.tolist()
            # preprocess texts here
            preprocessed_texts = [self.clean_text(text) for text in texts]
            vectorizer = TfidfVectorizer(stop_words='english')
            dtm = vectorizer.fit_transform(preprocessed_texts)
            lsa_model = TruncatedSVD(n_components=self.num_topics, random_state=42)
            lsa_model.fit(dtm)
            feature_names = vectorizer.get_feature_names()
            cluster_topics = []
            for topic_idx, topic in enumerate(lsa_model.components_):
                topic_words = [feature_names[i] for i in topic.argsort()[:-self.n_words_per_topic-1:-1]]
                cluster_topics.append(topic_words)
            topics_dict[cluster] = cluster_topics
        return topics_dict

    def add_lsa_topics_to_df(self, df, topics_dict):
        new_df = df.copy()
        new_df['topics'] = new_df[self.cluster].apply(lambda x: topics_dict[x])
        return new_df
