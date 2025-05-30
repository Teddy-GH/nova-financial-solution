from typing import Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

def setup_nltk():
	nltk.download("punkt")
	nltk.download("stopwords")
	nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str, do_lemmatize: bool = True) -> str:
	"""
	Clean text: lowercase, remove punctuation, stopwords, and lemmatize if needed.
	"""
	text = str(text).lower()
	text = re.sub(r"[^a-z\s]", "", text)
	tokens = text.split()
	filtered = [word for word in tokens if word not in stop_words]
	if do_lemmatize:
		filtered = [lemmatizer.lemmatize(word) for word in filtered]
	return " ".join(filtered)

def extract_sentiment(text: str) -> Optional[float]:
	"""
	Return polarity score from -1 (negative) to 1 (positive).
	"""
	try:
		return TextBlob(str(text)).sentiment.polarity
	except:
		return None