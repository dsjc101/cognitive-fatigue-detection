import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

# === Load data ===
X = np.load("data/X.npy")
y = np.load("data/y.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# === Load trained model ===
model = load_model("models/fatigue_lstm.h5")

# === Predict on validation set ===
y_pred_probs = model.predict(X_val)
y_pred = (y_pred_probs>=0.45).astype("int").reshape(-1)

# === Print metrics ===
print("\n Classification Report:")
print(classification_report(y_val, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fatigued"], yticklabels=["Normal", "Fatigued"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(" Confusion Matrix")
plt.tight_layout()
plt.show()