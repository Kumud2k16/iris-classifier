# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 2: Load the dataset
iris = load_iris()
X = iris.data      # Features (shape: 150 x 4)
y = iris.target    # Labels (shape: 150,)

# Step 3: Inspect data
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 samples:\n", X[:5])
print("First 5 labels:", y[:5])

# Step 4: Convert to DataFrame (for easier exploration)
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y
print("\nDataFrame shape:", df.shape)
print(df.head())

# Step 5: Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])