import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set a light, clean aesthetic for all plots
sns.set(style="whitegrid", palette="pastel")

# Load the dataset
data = pd.read_csv(r"C:\Users\lekha\Downloads\heart.xls")

# Basic overview of the data
print("First 5 rows of the dataset:\n", data.head())
print("\nDataset Info:\n")
print(data.info())
print("\nMissing values in each column:\n", data.isnull().sum())

# Separate the features (X) and target variable (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict and evaluate Decision Tree model
y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True, rounded=True)
plt.title("Decision Tree - Heart Disease Prediction", fontsize=16)
plt.show()

# Analyze overfitting by plotting accuracy vs tree depth
train_acc = []
test_acc = []
depth_range = range(1, 21)

for depth in depth_range:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(model.score(X_train, y_train))
    test_acc.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depth_range, train_acc, label="Train Accuracy", marker='o')
plt.plot(depth_range, test_acc, label="Test Accuracy", marker='s')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis: Decision Tree", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# Train and evaluate the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Visualize feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="Blues_d")
plt.title("Feature Importances - Random Forest", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Cross-validation scores for both models
cv_scores_dt = cross_val_score(dt_model, X, y, cv=5)
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5)

print("\nDecision Tree Cross-Validation Accuracy: %.2f%%" % (cv_scores_dt.mean() * 100))
print("Random Forest Cross-Validation Accuracy: %.2f%%" % (cv_scores_rf.mean() * 100))
