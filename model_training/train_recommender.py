import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path

print("Loading data...")
ratings = pd.read_csv("../data/ratings.csv")
products = pd.read_csv("../data/products.csv")

# Build user-item matrix
matrix = ratings.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

print("Computing user similarity...")
user_similarity = cosine_similarity(matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=matrix.index, columns=matrix.index)

# Save everything needed for prediction
model_data = {
    'user_similarity': user_similarity_df,
    'rating_matrix': matrix,
    'products_df': products.set_index('product_id')
}

Path("../models").mkdir(exist_ok=True)
joblib.dump(model_data, "../models/simple_recommender.pkl")
print("Simple recommender saved to models/simple_recommender.pkl")
print("Training complete! You can now run the app.")