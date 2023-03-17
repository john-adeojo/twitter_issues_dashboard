import re
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords


class TextCleaner:
    def __init__(self, stop_words=None, stop_words_remove=False):
        
        self.stop_words_remove = stop_words_remove
        if stop_words:
            self.stop_words = stop_words
        else:
            self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # convert input to str
        text = str(text)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove stop words
        if self.stop_words_remove == True:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        cleaned_text = text

        return cleaned_text
