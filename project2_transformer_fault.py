# ============================================================
# PROJECT 2: Power Transformer Fault Detection using DGA
# Tools: Python, scikit-learn, Pandas, Matplotlib, Seaborn
# Domain: Dissolved Gas Analysis (IEC 60599 standard)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── 1. Generate Realistic DGA Dataset ────────────────────
# Based on IEC 60599 typical fault gas ranges (ppm)
# Gases: H2, CH4, C2H2, C2H4, C2H6, CO, CO2
# Fault types: Normal, Thermal_Low, Thermal_High, Electrical, PD (Partial Discharge)

np.random.seed(42)
n = 300   # samples per class

def make_samples(n, h2, ch4, c2h2, c2h4, c2h6, co, co2, label, noise=0.15):
    data = {}
    for name, (lo, hi) in zip(
        ['H2','CH4','C2H2','C2H4','C2H6','CO','CO2'],
        [h2, ch4, c2h2, c2h4, c2h6, co, co2]
    ):
        base = np.random.uniform(lo, hi, n)
        data[name] = np.clip(base * (1 + np.random.normal(0, noise, n)), 0, None)
    data['Fault'] = label
    return pd.DataFrame(data)

df = pd.concat([
    make_samples(n, (10,50),   (5,30),    (0,1),    (5,20),   (2,10),  (50,200),  (500,1500), "Normal"),
    make_samples(n, (50,150),  (50,200),  (0,2),    (100,400),(20,80), (200,600), (1500,4000),"Thermal_Low"),
    make_samples(n, (100,300), (100,300), (1,5),    (300,800),(50,150),(400,1000),(3000,8000),"Thermal_High"),
    make_samples(n, (200,600), (30,100),  (50,200), (50,200), (10,40), (100,400), (1000,4000),"Electrical"),
    make_samples(n, (100,500), (10,50),   (1,10),   (5,30),   (2,15),  (50,200),  (500,2000), "PD"),
], ignore_index=True)

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df['Fault'].value_counts())

# ── 2. Save Dataset ───────────────────────────────────────
df.to_csv("transformer_dga_dataset.csv", index=False)
print("\nDataset saved as transformer_dga_dataset.csv")

# ── 3. Preprocessing ──────────────────────────────────────
le = LabelEncoder()
df['Fault_enc'] = le.fit_transform(df['Fault'])

features = ['H2','CH4','C2H2','C2H4','C2H6','CO','CO2']
X = df[features]
y = df['Fault_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train Models ───────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)

rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)

print("\n── Random Forest ──────────────────────────────")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

print("── Decision Tree ──────────────────────────────")
print(classification_report(y_test, dt_pred, target_names=le.classes_))

# ── 5. Confusion Matrix ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pred, title in zip(axes, [rf_pred, dt_pred],
                            ["Random Forest", "Decision Tree"]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_,
                cmap="Blues")
    ax.set_title(f"{title} – Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("transformer_fault_confusion.png", dpi=150)
plt.show()

# ── 6. Feature Importance ─────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 4))
importances.plot(kind='barh', color='steelblue')
plt.title("Feature Importance – Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("transformer_feature_importance.png", dpi=150)
plt.show()
print("Plots saved.")
