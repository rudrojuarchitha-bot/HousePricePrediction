import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("house_data.csv")

# Select input features
X = data[["area", "bedrooms", "bathrooms"]]

# Select the target value
y = data["price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "house_model.pkl")

print("House price prediction model trained successfully!")