
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import threading
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pynput import keyboard, mouse

# ==== App Setup ====
st.set_page_config(page_title="Fatigue Detector", layout="centered")
DATA_DIR = "data"
LABEL_FILE = os.path.join(DATA_DIR, "session_labels.csv")
GLOBAL_MODEL_PATH = "models/fatigue_lstm.h5"
os.makedirs(DATA_DIR, exist_ok=True)
import sys
import streamlit as st
st.write("🐍 Python version:", sys.version)
keyboard_log = []
mouse_log = []
last_key_time = None
last_mouse_pos = None
last_mouse_time = None

# ==== Keyboard Logger Thread ====
def run_keyboard_logger(listener_store):
    def on_press(key):
        global last_key_time
        now = time.time()
        delay = (now - last_key_time) if last_key_time else 0
        last_key_time = now
        keyboard_log.append({
            "key_hold_time": 0.05,
            "inter_key_delay": delay,
            "is_backspace": int(getattr(key, 'char', '') == '\b'),
            "timestamp": round(now, 3)
        })

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener_store["keyboard"] = listener

# ==== Mouse Logger Thread ====
def run_mouse_logger(listener_store):
    def on_move(x, y):
        global last_mouse_pos, last_mouse_time
        now = time.time()
        if last_mouse_pos and last_mouse_time:
            dx = x - last_mouse_pos[0]
            dy = y - last_mouse_pos[1]
            dist = (dx**2 + dy**2)**0.5
            dt = now - last_mouse_time
            speed = round(dist / dt, 4) if dt > 0 else 0
        else:
            speed = 0
        last_mouse_pos = (x, y)
        last_mouse_time = now
        mouse_log.append({
            "mouse_speed": speed,
            "click_event": 0,
            "timestamp": round(now, 3)
        })

    def on_click(x, y, button, pressed):
        if pressed:
            mouse_log.append({
                "mouse_speed": 0,
                "click_event": 1,
                "timestamp": round(time.time(), 3)
            })

    listener = mouse.Listener(on_move=on_move, on_click=on_click)
    listener.start()
    listener_store["mouse"] = listener

# ==== Merge and Save ====
def merge_and_save(user_id, session_name, save=True):
    df_kb = pd.DataFrame(keyboard_log)
    df_mouse = pd.DataFrame(mouse_log)
    df_kb["timestamp"] = df_kb["timestamp"].round(2)
    df_mouse["timestamp"] = df_mouse["timestamp"].round(2)
    df = pd.merge_asof(df_kb.sort_values("timestamp"), df_mouse.sort_values("timestamp"),
                       on="timestamp", direction='nearest', tolerance=0.1)
    df = df.fillna(0)
    df["session_time_of_day_x"] = datetime.now().hour / 24
    df["session_time_of_day_y"] = datetime.now().minute / 60
    df = df[["timestamp", "key_hold_time", "inter_key_delay", "is_backspace",
             "mouse_speed", "click_event", "session_time_of_day_x", "session_time_of_day_y"]]
    if save:
        df.to_csv(os.path.join(DATA_DIR, session_name), index=False)
    return df, session_name

# ==== Prediction ====
def predict_on_session(df, model):
    scaler = MinMaxScaler()
    features = df[["key_hold_time", "inter_key_delay", "is_backspace",
                   "mouse_speed", "click_event", "session_time_of_day_x", "session_time_of_day_y"]]
    X = scaler.fit_transform(features)
    X = np.expand_dims(X[:100], axis=0)
    pred = model.predict(X)
    return float(pred[0][0]), X.squeeze()

@st.cache_resource
def load_global_model():
    return load_model(GLOBAL_MODEL_PATH)

# ==== UI ====
st.title("Cognitive Fatigue Detection App")
user_id = st.text_input("Enter User ID", value="user1")
mode = st.selectbox("Choose Mode", ["Train", "Test"])
duration = st.selectbox(
    "Select Recording Duration",
    options=[120, 300, 600],
    format_func=lambda x: f"{x//60} minutes",
    index=0
)

if 'recording_complete' not in st.session_state:
    st.session_state.recording_complete = False
    st.session_state.recorded_df = None
    st.session_state.session_name = None
    st.session_state.label_choice = "Relaxed"

# === Train Mode ===
if mode == "Train":
    st.header("Record a Training Session")

    if not st.session_state.recording_complete:
        if st.button("Start Recording"):
            keyboard_log.clear()
            mouse_log.clear()
            st.info(f"Recording for {duration} seconds... Use keyboard and mouse naturally.")

            listener_store = {}
            t1 = threading.Thread(target=run_keyboard_logger, args=(listener_store,))
            t2 = threading.Thread(target=run_mouse_logger, args=(listener_store,))
            t1.start()
            t2.start()
            time.sleep(duration)
            listener_store["keyboard"].stop()
            listener_store["mouse"].stop()

            existing = [f for f in os.listdir(DATA_DIR) if f.startswith(f"session_{user_id}_s")]
            nums = [int(f.split("_s")[1].split(".")[0]) for f in existing if "_s" in f]
            next_num = max(nums, default=0) + 1
            session_name = f"session_{user_id}_s{next_num}.csv"

            df, _ = merge_and_save(user_id, session_name, save=False)
            st.session_state.recording_complete = True
            st.session_state.recorded_df = df
            st.session_state.session_name = session_name
            st.success("Recording complete. Choose label and click Save Session.")

    if st.session_state.recording_complete:
        st.write("Label this session:")
        st.session_state.label_choice = st.radio(
            "How was this session?", ["Relaxed", "Fatigued"], key="label_radio"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Session"):
                label_val = 0 if st.session_state.label_choice == "Relaxed" else 1
                df = st.session_state.recorded_df
                session_name = st.session_state.session_name
                df["label"] = label_val
                df["user_id"] = user_id
                df.to_csv(os.path.join(DATA_DIR, session_name), index=False)

                label_df = pd.DataFrame({
                    "session_file": [session_name],
                    "fatigue": [label_val]
                })
                if os.path.exists(LABEL_FILE):
                 with open(LABEL_FILE, "a", newline="\n") as f:
                   label_df.to_csv(f, header=False, index=False)
                else:
                     label_df.to_csv(LABEL_FILE, index=False)

                st.success(f"Session saved: {session_name} labeled as {st.session_state.label_choice}")
                st.session_state.recording_complete = False

        with col2:
            if st.button("Cancel"):
                st.session_state.recording_complete = False
                st.session_state.recorded_df = None
                st.session_state.session_name = None
                st.warning("Session discarded.")

# === Test Mode ===
elif mode == "Test":
    st.header("Test Mode — Real-Time Prediction")
    test_type = st.radio("Choose prediction mode:", ["Global", "Personal", "Hybrid"])
    if st.button("Start Test"):
        keyboard_log.clear()
        mouse_log.clear()
        st.info(f"Recording for {duration} seconds...")

        listener_store = {}
        t1 = threading.Thread(target=run_keyboard_logger, args=(listener_store,))
        t2 = threading.Thread(target=run_mouse_logger, args=(listener_store,))
        t1.start()
        t2.start()
        time.sleep(duration)
        listener_store["keyboard"].stop()
        listener_store["mouse"].stop()

        df, _ = merge_and_save(user_id, "test_session.csv", save=False)
        global_model = load_global_model()
        global_score, global_seq = predict_on_session(df, global_model)
        final_score = global_score
        show_seq = global_seq

        if test_type in ["Personal", "Hybrid"]:
            try:
                session_df = pd.read_csv(LABEL_FILE)
                session_df = session_df[session_df.session_file.str.contains(f"session_{user_id}_")]
                session_files = session_df.session_file.tolist()
                labels = session_df.fatigue.tolist()
                all_chunks = []
                for f in session_files:
                    d = pd.read_csv(os.path.join(DATA_DIR, f)).drop(columns=["timestamp", "label", "user_id"])
                    d = (d - d.min()) / (d.max() - d.min())
                    for i in range(0, len(d) - 100 + 1, 100):
                        all_chunks.append((d.iloc[i:i+100].values, labels[session_files.index(f)]))
                X = np.array([x[0] for x in all_chunks])
                y = np.array([x[1] for x in all_chunks])
                model = Sequential()
                model.add(LSTM(128, input_shape=(100, 7), return_sequences=False))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(optimizer='adam', loss='binary_crossentropy')
                model.fit(X, y, epochs=10, batch_size=8, verbose=0)
                personal_score, personal_seq = predict_on_session(df, model)
                if test_type == "Personal":
                    final_score = personal_score
                    show_seq = personal_seq
                elif test_type == "Hybrid":
                    final_score = (global_score + personal_score) / 2
                    show_seq = (global_seq + personal_seq) / 2
            except Exception as e:
                st.error("Personal model could not be built. Showing global result only.")

        st.subheader("Prediction Result")
        st.write(f"Fatigue Score: {final_score:.2f}")
        st.success("Fatigued" if final_score > 0.45 else "Relaxed")

        st.subheader("Feature Trend (First 100 Steps)")
        fig, ax = plt.subplots()
        ax.plot(show_seq)
        ax.set_title("Normalized Feature Sequence")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Feature Value")
        st.pyplot(fig)
