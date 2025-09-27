# ==============================================================================
# HOUSING PRICE PREDICTION: LINEAR REGRESSION
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SETUP AND DATA LOADING
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import kagglehub

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

# Download and load the dataset
print("Downloading the dataset...")
path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction")
file_path = f"{path}/Housing.csv"
df = pd.read_csv(file_path)
print("Dataset loaded successfully!")

# ------------------------------------------------------------------------------
# 2. DATA PREPROCESSING AND EXPLORATION
# ------------------------------------------------------------------------------
print("\n--- Initial Data ---")
print(df.head())
df.info()

# Convert categorical binary features to numeric
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))

print("\n--- Data after Preprocessing ---")
print(df.head())

# Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Housing Features')
plt.show()

# ------------------------------------------------------------------------------
# 3. SIMPLE LINEAR REGRESSION (Price vs. Area)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("3. SIMPLE LINEAR REGRESSION")
print("="*50 + "\n")

# Define features (X) and target (y)
X_simple = df[['area']]
y_simple = df['price']

# Split data
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Train the model
simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train_simple)

# Evaluate the model
y_pred_simple = simple_model.predict(X_test_simple)
print('--- Simple Linear Regression Evaluation ---')
print('R-squared (R²):', metrics.r2_score(y_test_simple, y_pred_simple))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test_simple, y_pred_simple)))
print(f"Coefficient (slope): {simple_model.coef_[0]}")

# Plotting the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual Prices')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Price vs. Area')
plt.xlabel('Area (sq. ft.)')
plt.ylabel('Price')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# 4. MULTIPLE LINEAR REGRESSION
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("4. MULTIPLE LINEAR REGRESSION")
print("="*50 + "\n")

# Select multiple features
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'airconditioning']
X_multi = df[features]
y_multi = df['price']

# Split data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Train the model
multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)

# Evaluate the model
y_pred_multi = multi_model.predict(X_test_multi)
print('--- Multiple Linear Regression Evaluation ---')
print('R-squared (R²):', metrics.r2_score(y_test_multi, y_pred_multi))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test_multi, y_pred_multi)))

# Interpret coefficients
coeffs = pd.DataFrame(multi_model.coef_, X_multi.columns, columns=['Coefficient'])
print("\n--- Model Coefficients ---")
print(coeffs)

print("\nAnalysis complete.")
