import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
books = pd.read_csv('books.csv')

# Combine relevant features into a single string (author + genre)
books['Features'] = books['Author'] + ' ' + books['Genre']

# Use TF-IDF vectorization to transform text data into numeric vectors
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(books['Features'])

# Compute cosine similarity between all books
cosine_sim = cosine_similarity(feature_matrix)

def recommend_books(book_title, num_recommendations=3):
    # Find the index of the input book
    book_index = books[books['Title'] == book_title].index[0]
    
    # Get similarity scores for the input book with all other books
    similarity_scores = list(enumerate(cosine_sim[book_index]))
    
    # Sort the books by similarity score in descending order
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the most similar books (excluding the input book itself)
    recommended_indices = [i[0] for i in sorted_scores[1:num_recommendations+1]]
    
    # Return the recommended book titles
    return books['Title'].iloc[recommended_indices].tolist()

# Example usage
book_to_search = "1984"
recommendations = recommend_books(book_to_search)
print(f"Books similar to '{book_to_search}':")
for book in recommendations:
    print(f"- {book}")
