import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)

data = {
    'review': [
        "This product is amazing! I love it.",
        "Terrible experience, will never buy again.",
        "Okayish. Nothing special, but not bad either."
    ]
}
df = pd.DataFrame(data)