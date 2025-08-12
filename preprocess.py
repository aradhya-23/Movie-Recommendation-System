import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
movies = pd.read_csv("movies.csv")

# Fill missing values
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')

# Combine text features
movies['combined'] = movies['overview'] + " " + movies['genres']

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined'])

# Save processed files
with open("movies.pkl", "wb") as f:
    pickle.dump(movies, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Preprocessing complete — files saved: movies.pkl, tfidf.pkl, vectorizer.pkl")