"""
Titanic Survival: EDA + Baseline Model
========================================
A quick, self-contained data science project that analyses the Titanic dataset with a 79% accuracy from the features age, family size, number of embarked passagers, is alone and  :
1. Load data
2. Exploratory data analysis (with saved plots)
3. Feature engineering
4. Baseline logistic regression model
5. Evaluation

Run with: python notebooks/analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

sns.set_theme(style="whitegrid")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/titanic.csv")
print("Shape:", df.shape)
print(df.head())
print(df.isna().sum())

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=df, x="Survived", ax=axes[0])
axes[0].set_title("Survival Counts")
sns.barplot(data=df, x="Pclass", y="Survived", ax=axes[1])
axes[1].set_title("Survival Rate by Passenger Class")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/survival_overview.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Sex", y="Survived", ax=axes[0])
axes[0].set_title("Survival Rate by Sex")
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, ax=axes[1])
axes[1].set_title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/survival_by_sex_age.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
data = df.copy()
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)

features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone",
            "Embarked_Q", "Embarked_S"]
X = data[features]
y = data["Survived"]

# ---------------------------------------------------------------------------
# 4. Train / evaluate baseline model
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
preds = model.predict(X_test_s)

acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.3f}\n")
print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.title(f"Confusion Matrix (Accuracy={acc:.2f})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance (coefficients)
coef_df = pd.DataFrame({"feature": features, "coefficient": model.coef_[0]})
coef_df = coef_df.sort_values("coefficient")
plt.figure(figsize=(6, 4))
sns.barplot(data=coef_df, x="coefficient", y="feature", palette="viridis")
plt.title("Logistic Regression Coefficients")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
plt.close()

print(f"\nPlots saved to {OUT_DIR}/")
