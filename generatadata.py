import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate data
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'SciFi', 'Romance', 'Thriller']
data = {
    'Genre': np.random.choice(genres, n_samples),
    'Runtime': np.random.randint(80, 180, n_samples),
    'UserAge': np.random.randint(18, 70, n_samples),
}

# Create more realistic relationships
data['Liked'] = (
    (data['Genre'] == 'Action') & (data['UserAge'] < 30) |
    (data['Genre'] == 'Romance') & (data['UserAge'] > 40) |
    (data['Genre'] == 'SciFi') & (data['Runtime'] > 120) |
    (data['Genre'] == 'Comedy') & (data['Runtime'] < 110) |
    (np.random.rand(n_samples) < 0.5)  # Add some randomness
).astype(int)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("movie_preferences_large.csv", index=False)

print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nLiked distribution:\n{df['Liked'].value_counts(normalize=True)}")
print(f"\nGenre distribution:\n{df['Genre'].value_counts(normalize=True)}")

