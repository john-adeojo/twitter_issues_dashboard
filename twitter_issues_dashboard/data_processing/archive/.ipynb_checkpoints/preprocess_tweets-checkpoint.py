import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class TextCleaner():
    def __init__(self, stop_words_remove=True):
        self.stop_words_remove = stop_words_remove
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

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

        # Tokenize the text and convert to lowercase
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]

        # Remove stop words
        if self.stop_words_remove == True:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]

        # Lemmatize the tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Join the cleaned tokens back into a string
        cleaned_text = ' '.join(tokens)

        return cleaned_text
