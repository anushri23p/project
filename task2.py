import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv(r"C:\Users\ARATI\Desktop\internship\disney_movies.csv")

print("Dataset Loaded Successfully")

# Preprocessing
movies.fillna("", inplace=True)

text_columns = movies.select_dtypes(include="object").columns
movies["combined_features"] = movies[text_columns].apply(
    lambda row: " ".join(row.values.astype(str)),
    axis=1
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

# Cosine similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(movie_title, top_n=5):
    if movie_title not in movies.iloc[:, 0].values:
        print("Movie not found")
        return

    idx = movies[movies.iloc[:, 0] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    print(f"\nMovies similar to '{movie_title}':")
    for i in scores:
        print(movies.iloc[i[0], 0])

# Example
recommend(movies.iloc[0, 0])