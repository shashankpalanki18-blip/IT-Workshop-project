import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the dataset
print("Loading data...")
try:
    df = pd.read_csv('data/dataset.csv')
except FileNotFoundError:
    print("Error: Could not find 'data/dataset.csv'. Make sure you ran collect_data.py first!")
    exit()

# 2. Separate features (coordinates) and labels (letters)
y = df['label']
X = df.drop('label', axis=1)

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Classifier
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test the model's accuracy on the 20% of data it hasn't seen yet
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to the data folder
with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model successfully saved to 'data/model.pkl'")