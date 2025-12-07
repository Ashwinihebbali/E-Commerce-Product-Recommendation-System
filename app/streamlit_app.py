import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------- Load data & model ----------
@st.cache_data
def load_data():
    products = pd.read_csv("../data/products.csv")
    ratings = pd.read_csv("../data/ratings.csv")
    return products, ratings

@st.cache_resource
def load_recommender():
    return joblib.load("../models/simple_recommender.pkl")

products, ratings = load_data()
model_data = load_recommender()

user_similarity = model_data['user_similarity']
rating_matrix = model_data['rating_matrix']
products_df = model_data['products_df']

# ---------- App ----------
st.set_page_config(page_title="E-Shop Recommender", layout="wide")
st.title("E-Commerce Product Recommender")

page = st.sidebar.selectbox("Menu", ["Browse Products", "Get Recommendations"])

if page == "Browse Products":
    st.header("All Products")
    search = st.text_input("Search product name")
    cat = st.selectbox("Category", ["All"] + sorted(products['category'].unique()))
    
    df = products.copy()
    if search:
        df = df[df['name'].str.contains(search, case=False)]
    if cat != "All":
        df = df[df['category'] == cat]
    
    st.dataframe(df[['name', 'category', 'price']], use_container_width=True)

else:
    st.header("Your Personalized Recommendations")
    user_id = st.number_input("Enter your User ID (1-1000)", min_value=1, max_value=1000, value=42)
    
    if st.button("Generate Recommendations"):
        if user_id not in user_similarity.index:
            st.warning("New user! Showing popular items...")
            popular = ratings.groupby('product_id')['rating'].mean().sort_values(ascending=False).head(10)
        else:
            # Find similar users
            similar_users = user_similarity[user_id].sort_values(ascending=False)[1:11]
            # Weighted average of their ratings
            weighted_ratings = np.zeros(len(rating_matrix.columns))
            total_similarity = 0
            for sim_user, sim_score in similar_users.items():
                if sim_score > 0:
                    weighted_ratings += sim_score * rating_matrix.loc[sim_user]
                    total_similarity += sim_score
            if total_similarity > 0:
                scores = weighted_ratings / total_similarity
            else:
                scores = rating_matrix.mean(axis=0).values
            
            scores_series = pd.Series(scores, index=rating_matrix.columns)
            # Remove already rated items
            user_rated = ratings[ratings['user_id'] == user_id]['product_id']
            scores_series = scores_series.drop(user_rated, errors='ignore')
            popular = scores_series.sort_values(ascending=False).head(10)
        
        # Show results
        recs = products_df.loc[popular.index][['name', 'category', 'price']]
        recs['Predicted Score'] = popular.values
        st.dataframe(recs.reset_index(drop=True), use_container_width=True)