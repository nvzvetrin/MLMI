# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# Set Seaborn theme for attractive plots
sns.set_theme(style="darkgrid")  # More reliable than plt.style.use

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Rename 'target' to 'species' for clarity

# Convert target to categorical for better visualization
df['species'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Display first few rows of the dataset
print(df.head())

# 🔹 Pairplot with KDE and modern color theme
sns.pairplot(df, hue='species', palette="coolwarm", diag_kind='kde', markers=["o", "s", "D"])
plt.suptitle("Feature Relationships in Iris Dataset", y=1.02, fontsize=16, fontweight='bold')
plt.show()

# Split the dataset into features and target
X = df.drop('species', axis=1)
y = df['species']

# Convert categorical labels to numerical format
y = y.map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize Random Forest Classifier with optimized parameters
rf_model = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.3f}')

# 🔹 Confusion Matrix with Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="magma", linewidths=1, 
            xticklabels=iris.target_names, yticklabels=iris.target_names, cbar=False)

plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('📊 Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
plt.show()

# 🔹 Feature Importance Analysis
feature_importance_values = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importance_values)[::-1]
sorted_features = [X.columns[i] for i in sorted_indices]

# 🔹 Attractive Barplot for Feature Importance
plt.figure(figsize=(10, 6))
colors = sns.color_palette("coolwarm", len(sorted_features))  # Gradient color effect

sns.barplot(x=sorted_features, y=feature_importance_values[sorted_indices], palette=colors)

plt.xlabel("Feature", fontsize=12, fontweight='bold')
plt.ylabel("Importance Score", fontsize=12, fontweight='bold')
plt.title("🔥 Feature Importance in Random Forest Model", fontsize=14, fontweight='bold')
plt.xticks(rotation=30, fontsize=10)
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.show()
