"""
Text preprocessing module for emotion detection
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Class for preprocessing text data"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        original_text = text.strip()
        if not original_text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        filtered_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        # Join back to text
        cleaned_text = ' '.join(filtered_tokens)
        
        # If cleaning results in empty text, return basic cleaned version
        if not cleaned_text.strip():
            # Fallback: just remove punctuation and convert to lowercase
            fallback = re.sub(r'[^\w\s]', '', original_text.lower())
            return fallback.strip()
        
        return cleaned_text.strip()
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts
        
        Args:
            texts (list): List of texts to preprocess
            
        Returns:
            list: List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]
