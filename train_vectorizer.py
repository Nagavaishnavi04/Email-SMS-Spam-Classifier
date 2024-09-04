from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Example training data (replace with your actual data)
documents = [
    "This is a sample document.",
    "This document is another sample.",
    "And this is yet another sample document."
]

# Initialize and fit the TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(documents)  # Fitting the vectorizer to the training data

# Save the fitted vectorizer to a file
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
