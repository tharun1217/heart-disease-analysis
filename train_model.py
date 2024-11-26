# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load the dataset
data_path = r"D:\myprojects\Heart_Analysis\Heart_record.csv"
data = pd.read_csv(data_path)
print("Dataset Loaded Successfully!\n")
print("First 5 rows of the dataset:\n", data.head())

# Step 2: Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")
print("Data Summary:\n", data.describe())
print("\nNull Values in the Dataset:\n", data.isnull().sum())

# Visualize the class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='DEATH_EVENT', data=data, palette='Set2')
plt.title("Class Distribution")
plt.xlabel("Death Event (0: No, 1: Yes)")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Step 3: Feature Scaling and Data Splitting
X = data.drop(columns=["DEATH_EVENT"])
y = data["DEATH_EVENT"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets!")

# Step 4: Model Training
print("\n--- Model Training ---")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed!")

# Step 5: Model Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluation Metrics
print("\n--- Model Evaluation ---")
print("\nTrain Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_pred_test))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
feature_importances = pd.DataFrame(model.feature_importances_, index=data.columns[:-1], columns=["Importance"]).sort_values(by="Importance", ascending=False)
print("\nFeature Importance:\n", feature_importances)

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances.Importance, y=feature_importances.index, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Step 6: Save model using pickle
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and Scaler saved as 'heart_disease_model.pkl' and 'scaler.pkl'")
