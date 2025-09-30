import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib  # for saving the model

# 1️⃣ Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2️⃣ Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4️⃣ Make predictions
y_pred = model.predict(X_test)

# 5️⃣ Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 6️⃣ Create outputs folder if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 7️⃣ Save the trained model
model_path = os.path.join(output_dir, "model.joblib")
joblib.dump(model, model_path)
print(f"Trained model saved to: {model_path}")

# 8️⃣ Save confusion matrix as PNG
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(iris.target_names)), iris.target_names)
plt.yticks(range(len(iris.target_names)), iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Annotate each cell
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()
print(f"Confusion matrix saved to: {conf_matrix_path}")
