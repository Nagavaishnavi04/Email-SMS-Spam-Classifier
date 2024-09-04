from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample training data (replace this with your actual training data)
documents = [
    "This is a sample document.",
    "This document is another sample.",
    "And this is yet another sample document."
]

# Example labels for training (replace with your actual labels, 0 = not spam, 1 = spam)
labels = [0, 1, 0]

# Initialize and fit the TfidfVectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(documents)  # Fit and transform the training data

# Initialize and train the MultinomialNB model
model = MultinomialNB()
model.fit(X_train, labels)

# Save the fitted vectorizer and trained model as .pkl files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
