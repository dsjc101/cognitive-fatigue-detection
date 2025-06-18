from pynput import mouse
import pandas as pd
import time
import os
from datetime import datetime

log = []
last_position = None
last_time = None

def get_time_of_day():
    now = datetime.now()
    return round((now.hour * 3600 + now.minute * 60 + now.second) / (24 * 3600), 4)

def on_move(x, y):
    global last_position, last_time
    now = time.time()
    session_time = get_time_of_day()

    if last_position and last_time:
        dx = x - last_position[0]
        dy = y - last_position[1]
        distance = (dx**2 + dy**2)**0.5
        dt = now - last_time

        speed = round(distance / dt, 4) if dt > 0 else 0
    else:
        speed = 0

    last_position = (x, y)
    last_time = now

    log.append({
        "event": "move",
        "mouse_speed": speed,
        "x": x,
        "y": y,
        "time_of_day": session_time,
        "timestamp": round(now, 3)
    })

def on_click(x, y, button, pressed):
    if pressed:
        session_time = get_time_of_day()
        log.append({
            "event": "click",
            "x": x,
            "y": y,
            "mouse_speed": None,
            "time_of_day": session_time,
            "timestamp": round(time.time(), 3)
        })

def save_mouse_log(user="user1", session="s1"):
    df = pd.DataFrame(log)
    df["user_id"] = user
    df["session_id"] = session

    #  Clean: Remove move events with unrealistic speed > 1000 px/s
    df = df[
        (df["event"] != "move") | (df["mouse_speed"].fillna(0) < 1000)
    ]

    os.makedirs("data", exist_ok=True)
    filename = f"data/mouse_{user}_{session}.csv"
    df.to_csv(filename, index=False)
    print(f"\n Mouse log saved to {filename}")

def run_mouse_logger():
    user = input("Enter user ID: ") or "user1"
    session = input("Enter session ID: ") or "s1"
    print(" Tracking mouse. Press Ctrl + C in terminal to stop and save.\n")

    listener = mouse.Listener(on_move=on_move, on_click=on_click)
    listener.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        listener.stop()
        save_mouse_log(user, session)

if __name__ == "__main__":
    run_mouse_logger()
