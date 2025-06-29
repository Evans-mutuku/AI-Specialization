# Step 1: Install and import necessary libraries
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

data = {
    'review': [
        "This product is amazing! I love it.",
        "Terrible experience, will never buy again.",
        "Okayish. Nothing special, but not bad either."
    ]
}
df = pd.DataFrame(data)

def preprocess_text(text):
    try:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        return f"Error: {e}"


df['cleaned_review'] = df['review'].apply(preprocess_text)
print(df[['review', 'cleaned_review']])