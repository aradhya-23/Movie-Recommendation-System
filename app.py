import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Load saved files
with open("movies.pkl", "rb") as f:
    movies = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)


# Function to get poster from TMDB
def fetch_poster(movie_name):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(search_url)
    data = response.json()
    if data['results']:
        poster_path = data['results'][0]['poster_path']
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Recommendation function
def recommend(movie_title, num_recommendations=5):
    if movie_title not in movies['title'].values:
        return []
    
    idx = movies.index[movies['title'] == movie_title][0]
    movie_vector = tfidf_matrix[idx]
    
    sim_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:num_recommendations+1]
    
    rec_movies = movies['title'].iloc[sim_indices].tolist()
    rec_posters = [fetch_poster(title) for title in rec_movies]
    return list(zip(rec_movies, rec_posters))

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")

movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie you like:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        cols = st.columns(5)
        for col, (title, poster) in zip(cols, recommendations):
            with col:
                st.text(title)
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.write("No poster found")
