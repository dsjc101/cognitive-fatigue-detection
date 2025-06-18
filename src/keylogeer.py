from pynput import keyboard
import pandas as pd
import time
import os
from datetime import datetime

log = []
pressed_keys = {}
last_press_time = None

# Converts current time to float between 0 and 1
def time_of_day_float():
    now = datetime.now()
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    return round(seconds / 86400, 4)

def on_press(key):
    global last_press_time
    now = time.time()
    delay = now - last_press_time if last_press_time else 0
    last_press_time = now

    try:
        k = key.char
    except:
        k = str(key)

    is_backspace = 1 if k == 'Key.backspace' else 0
    time_of_day = time_of_day_float()
    pressed_keys[k] = now

    log.append({
        "event": "press",
        "key": k,
        "inter_key_delay": round(delay, 3),
        "is_backspace": is_backspace,
        "session_time_of_day": time_of_day,
        "timestamp": round(now, 3)
    })

def on_release(key):
    now = time.time()
    try:
        k = key.char
    except:
        k = str(key)

    hold = now - pressed_keys.get(k, now)

    log.append({
        "event": "release",
        "key": k,
        "key_hold_time": round(hold, 3),
        "timestamp": round(now, 3)
    })

    if key == keyboard.Key.esc:
        return False  # Stop on ESC

def run_logger():
    user = input("Enter user ID: ") or "user1"
    session = input("Enter session ID: ") or "s1"
    print("Typing... Press ESC to stop and save.\n")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    df = pd.DataFrame(log)
    df["user_id"] = user
    df["session_id"] = session

    os.makedirs("data", exist_ok=True)
    filename = f"data/keylog_{user}_{session}.csv"
    df.to_csv(filename, index=False)
    print(f"\n Data saved to {filename}")

if __name__ == "__main__":
    run_logger()
