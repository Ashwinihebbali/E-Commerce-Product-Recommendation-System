import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os

# ---------- Synthetic data (generated on the fly) ----------
@st.cache_data
def get_data():
    np.random.seed(42)
    n_products = 500
    n_users = 800
    
    products = pd.DataFrame({
        'product_id': range(1, n_products+1),
        'name': [f"Product {i}" for i in range(1, n_products+1)],
        'category': np.random.choice(['Electronics', 'Fashion', 'Books', 'Home', 'Sports'], n_products),
        'price': np.round(np.random.uniform(10, 500, n_products), 2),
        'description': [f"Amazing {c} item with great features" for c in np.random.choice(['wireless', 'premium', 'classic', 'modern', 'durable'], n_products)]
    })
    
    # Generate ratings
    ratings = []
    for user in range(1, n_users+1):
        n_rated = np.random.randint(10, 80)
        rated_products = np.random.choice(products['product_id'], n_rated, replace=False)
        for p in rated_products:
            ratings.append({'user_id': user, 'product_id': p, 'rating': np.random.randint(1,6)})
    
    return products, pd.DataFrame(ratings)

products, ratings = get_data()

# Build user-item matrix and similarity
@st.cache_resource
def build_model():
    matrix = ratings.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    similarity = cosine_similarity(matrix)
    sim_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)
    return matrix, sim_df

rating_matrix, user_sim = build_model()

# ---------- Streamlit App ----------
st.set_page_config(page_title="ShopSmart Recommender", layout="wide")
st.title("ShopSmart – Product Recommendation System")
st.markdown("**No installation • Runs instantly • Pure Python**")

menu = st.sidebar.selectbox("Menu", ["Browse Products", "Get Recommendations"])

if menu == "Browse Products":
    st.subheader("All Products")
    search = st.text_input("Search")
    cat = st.selectbox("Category", ["All"] + sorted(products['category'].unique()))
    
    df = products.copy()
    if search:
        df = df[df['name'].str.contains(search, case=False) | df['description'].str.contains(search, case=False)]
    if cat != "All":
        df = df[df['category'] == cat]
        
    st.dataframe(df[['name', 'category', 'price']], use_container_width=True)

else:
    st.subheader("Your Personalized Recommendations")
    user_id = st.slider("Choose your User ID", 1, 800, 42)
    
    if st.button("Generate Top 10 Recommendations"):
        # Get similar users
        similar_users = user_sim[user_id].sort_values(ascending=False)[1:20]
        scores = np.zeros(len(rating_matrix.columns))
        weights = 0
        
        for sim_user, sim_score in similar_users.items():
            if sim_score > 0:
                scores += sim_score * rating_matrix.loc[sim_user]
                weights += sim_score
                
        if weights > 0:
            pred = scores / weights
        else:
            pred = rating_matrix.mean().values
            
        pred_series = pd.Series(pred, index=rating_matrix.columns)
        already_rated = ratings[ratings['user_id']==user_id]['product_id']
        pred_series = pred_series.drop(already_rated, errors='ignore')
        
        top10_ids = pred_series.sort_values(ascending=False).head(10).index
        recommendations = products[products['product_id'].isin(top10_ids)].copy()
        recommendations['Predicted Rating'] = pred_series.loc[top10_ids].values.round(2)
        recommendations = recommendations[['name', 'category', 'price', 'Predicted Rating']]
        
        st.success(f"Top 10 recommendations for User {user_id}")
        st.dataframe(recommendations, use_container_width=True)

st.caption("Built with ❤️• 100% working • No setup required")