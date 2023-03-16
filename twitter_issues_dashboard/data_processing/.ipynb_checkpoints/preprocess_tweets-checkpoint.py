import re
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords


class TextCleaner:
    def __init__(self, stop_words=None):
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

        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)

        # Remove ASCII characters
        text = re.sub(r'[^\x00-\x7f]', '', text)

        # Remove @ symbols and # symbols
        text = re.sub(r'[@#]\w+', '', text)

        # Remove variations of "royal mail"
        text = re.sub(r'royal\s*mail', '', text, flags=re.IGNORECASE)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stop words
        tokens = [token for token in tokens if token.lower() not in self.stop_words]

        # Lemmatize the tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Join the cleaned tokens back into a string
        cleaned_text = ' '.join(tokens)

        return cleaned_text
