import os,time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#  Load data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

print(f" Loaded X shape: {X.shape}, y shape: {y.shape}")

#  Split data 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# === Build LSTM model ===
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
#model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary output

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# === Train ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
cw = dict(enumerate(class_weights))

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    class_weight=cw,
    verbose=1
)

# === Save the model ===
model.save("models/fatigue_lstm.h5")
print(" Model saved to models/fatigue_lstm.h5")
# Confirm overwrite
print(" Model last updated at:", time.ctime(os.path.getmtime("models/fatigue_lstm.h5")))

import matplotlib.pyplot as plt

y_pred_prob = model.predict(X_val)
plt.hist(y_pred_prob, bins=20)
plt.title("Distribution of predicted scores")
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()

