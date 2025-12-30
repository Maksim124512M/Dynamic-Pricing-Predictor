import numpy as np
import pandas as pd


n = 5000

df = pd.DataFrame({
    'product_id': np.arange(n),
    'category': np.random.choice(a=['electronics', 'home'], size=n),
    'price': np.random.randint(20, 4000, size=n),
    'rating': np.random.randint(1, 5, size=n),
    'reviews': np.random.randint(0, 400, size=n),
    'discount': np.random.randint(0, 70, size=n),
    'sales_last_7d': np.random.randint(0, 80, size=n),
})

df['revenue_last_7d'] = (df['price'] - (df['price'] / 100 * 20)) * df['sales_last_7d']
df['revenue_next_7d'] = (df['price'] * (1 - df['discount']/100) * df['sales_last_7d']) \
                        * (1 + np.random.uniform(-0.2, 0.2, size=len(df)))

df.to_csv('data/products.csv', index=False)