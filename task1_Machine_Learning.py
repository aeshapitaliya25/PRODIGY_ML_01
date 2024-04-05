import numpy as np

# Sample data (Square footage, number of bedrooms, number of bathrooms, price)
data = np.array([
    [1500, 3, 2, 300000],
    [2000, 4, 3, 400000],
    [1200, 2, 1, 250000],
    [1800, 3, 2, 350000],
    [2100, 4, 3, 450000],
    [1000, 2, 1, 200000]
])

# Splitting the data into features and target variable
X = data[:, :-1]  # Features (Square footage, number of bedrooms, number of bathrooms)
y = data[:, -1]   # Target variable (Price)

# Adding a column of ones for the intercept term
X_with_intercept = np.c_[np.ones(X.shape[0]), X]

# Calculating the parameters using the normal equation (θ = (X^T * X)^-1 * X^T * y)
theta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

def predict_price(square_footage, num_bedrooms, num_bathrooms):
    # Predicting house prices for new data
    new_house = np.array([[1, square_footage, num_bedrooms, num_bathrooms]])  # Intercept term, square footage, number of bedrooms, number of bathrooms
    predicted_price = new_house @ theta
    return predicted_price[0]

# Input square footage, number of bedrooms, and number of bathrooms
square_footage = float(input("Enter the square footage of the house: "))
num_bedrooms = int(input("Enter the number of bedrooms: "))
num_bathrooms = int(input("Enter the number of bathrooms: "))

# Predicting the price
predicted_price = predict_price(square_footage, num_bedrooms, num_bathrooms)
print("Predicted price for the house: $", predicted_price)