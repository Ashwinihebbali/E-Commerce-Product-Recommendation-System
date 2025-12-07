import pandas as pd
import numpy as np
import random

# Fixed categories and word lists for synthetic names/descriptions
categories = ['Electronics', 'Books', 'Clothing', 'Home & Garden', 'Sports']
nouns = ['Device', 'Book', 'Shirt', 'Tool', 'Shoe', 'Phone', 'Novel', 'Jacket', 'Lamp', 'Ball']
adjs = ['Wireless', 'Classic', 'Cotton', 'Garden', 'Running', 'Smart', 'Mystery', 'Leather', 'Desk', 'Soccer']

# Generate products
products = []
for i in range(1000):
    cat = random.choice(categories)
    name = random.choice(adjs) + ' ' + random.choice(nouns)
    desc_words = [random.choice(adjs + nouns) for _ in range(10)]
    desc = ' '.join(desc_words) + '.'
    price = round(random.uniform(10.0, 500.0), 2)
    products.append({
        'product_id': i + 1,
        'name': name,
        'category': cat,
        'description': desc,
        'price': price
    })

products_df = pd.DataFrame(products)
products_df.to_csv('data/products.csv', index=False)
print(f"Generated {len(products_df)} products in data/products.csv")

# Generate ratings
ratings = []
num_users = 1000
num_ratings_per_user = 50

for user_id in range(1, num_users + 1):
    num_ratings = random.randint(20, num_ratings_per_user)
    rated_products = random.sample(range(1, 1001), num_ratings)
    for prod_id in rated_products:
        rating = random.randint(1, 5)
        ratings.append({
            'user_id': user_id,
            'product_id': prod_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings)
ratings_df.to_csv('data/ratings.csv', index=False)
print(f"Generated {len(ratings_df)} ratings in data/ratings.csv")