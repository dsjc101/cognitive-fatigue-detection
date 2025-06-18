import pandas as pd
import numpy as np

# === CONFIGURATION ===
TIME_STEPS = 100
label_file = "data/session_labels.csv"

# === Load Labels ===
labels_df = pd.read_csv(label_file)
X_data = []
y_data = []

for i in range(len(labels_df)):
    session_file = labels_df.loc[i, "session_file"]
    label = labels_df.loc[i, "fatigue"]
    df = pd.read_csv(f"data/{session_file}")

    # Keep only the same 7 features used in prediction
    df = df[["key_hold_time", "inter_key_delay", "is_backspace",
         "mouse_speed", "click_event", "session_time_of_day_x", "session_time_of_day_y"]]

    # Normalize per session (0 to 1)
    df = (df - df.min()) / (df.max() - df.min())
    df = df.fillna(0)

    # Slice into chunks of fixed time steps
    for j in range(0, len(df) - TIME_STEPS + 1, TIME_STEPS):
        chunk = df.iloc[j:j + TIME_STEPS].values
        X_data.append(chunk)
        y_data.append(label)

# === Convert to numpy ===
X = np.array(X_data)
y = np.array(y_data)

print(f"\n Done! Shape of X: {X.shape}, y: {y.shape}")

# === Save to .npy ===
np.save("data/X.npy", X)
np.save("data/y.npy", y)
print(" Saved: data/X.npy and data/y.npy")