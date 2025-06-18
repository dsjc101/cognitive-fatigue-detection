import pandas as pd

# CHANGE these to your actual session file names
# Ask user for input (interactive mode)
user = input("Enter user ID: ") or "user1"
session = input("Enter session ID: ") or "s1"

# Load both CSVs
key_df = pd.read_csv(f"data/keylog_{user}_{session}.csv")
mouse_df = pd.read_csv(f"data/mouse_{user}_{session}.csv")

# --- Clean Keyboard Data ---

# Keep only 'press' events and attach matching hold times
press_df = key_df[key_df["event"] == "press"].copy()
release_df = key_df[key_df["event"] == "release"].copy()

# Use reset_index to align release rows with press rows
press_df = press_df.reset_index(drop=True)
release_df = release_df.reset_index(drop=True)

# Add key_hold_time to press rows
press_df["key_hold_time"] = release_df["key_hold_time"]

# Select relevant keyboard features
keyboard_final = press_df[[
    "timestamp", "key_hold_time", "inter_key_delay", "is_backspace", "session_time_of_day"
]].copy()

# --- Clean Mouse Data ---

# Convert click into 1/0
mouse_df["click_event"] = (mouse_df["event"] == "click").astype(int)

# Keep only needed features
mouse_final = mouse_df[[
    "timestamp", "mouse_speed", "click_event", "time_of_day"
]].copy()
mouse_final.rename(columns={"time_of_day": "session_time_of_day"}, inplace=True)

# --- Merge both based on timestamp (approximate join) ---

# Round timestamps to 2 decimal places (to align better)
keyboard_final["timestamp"] = keyboard_final["timestamp"].round(2)
mouse_final["timestamp"] = mouse_final["timestamp"].round(2)

# Merge using outer join to keep all events
merged = pd.merge(keyboard_final, mouse_final, on="timestamp", how="outer")

# Sort by time
merged = merged.sort_values("timestamp").reset_index(drop=True)

# Fill any missing values with 0 (safe default)
merged = merged.fillna(0)

# Drop rows where everything is 0 (no activity from mouse or keyboard)
merged = merged[~((merged["key_hold_time"] == 0) & 
                  (merged["inter_key_delay"] == 0) & 
                  (merged["mouse_speed"] == 0) & 
                  (merged["click_event"] == 0))]

# --- Save merged DataFrame ---
merged.to_csv(f"data/session_{user}_{session}.csv", index=False)
print(f" Merged session saved to: data/session_{user}_{session}.csv")
