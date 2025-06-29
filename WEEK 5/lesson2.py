# ✅ Step 1: Import libraries
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# ✅ Step 2: Force download of NLTK resources to prevent LookupError
nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)

# ✅ Step 3: Sample DataFrame
data = {
    'review': [
        "This product is amazing! I love it.",
        "Terrible experience, will never buy again.",
        "Okayish. Nothing special, but not bad either."
    ]
}
df = pd.DataFrame(data)