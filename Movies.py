import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:/Users/zlati/Desktop/Ml Projects/Movie LikeDislike/movie_preferences_large.csv")

X = data[['Genre', 'Runtime', 'UserAge']]
y = data['Liked']

X = pd.get_dummies(X, columns=['Genre'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Model accuracy: {accuracy:.2f}")


def predict_movie_preference(genre, runtime, user_age):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Genre': [genre],
        'Runtime': [runtime],
        'UserAge': [user_age]
    })

      # One-hot encode the genre
    input_encoded = pd.get_dummies(input_data, columns=['Genre'])
    
    # Ensure all genre columns from training are present
    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match the training data
    input_encoded = input_encoded[X.columns]
    
    # Make prediction
    prediction = clf.predict(input_encoded)
    
    return "Liked" if prediction[0] == 1 else "Not Liked"

# Interactive input
user_genre = input("Enter a movie genre: ")
user_runtime = int(input("Enter the movie runtime in minutes: "))
user_age = int(input("Enter the user's age: "))

result = predict_movie_preference(user_genre, user_runtime, user_age)
print(f"The user is predicted to {result} this movie.")