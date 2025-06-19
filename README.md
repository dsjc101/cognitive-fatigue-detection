#  Cognitive Fatigue Detection via Keyboard & Mouse Dynamics

A real-time behavioral biometrics system that detects cognitive fatigue based on user interaction patterns — leveraging keystroke timing, cursor movement, and click behavior. This project combines LSTM-based sequence modeling with personalized and global inference to support proactive mental wellness tracking.

# Project Overview

This project captures and analyzes user keyboard and mouse activity to detect signs of cognitive fatigue. A custom Streamlit app allows users to record their sessions, label their mental state, and receive real-time predictions powered by a trained LSTM model.

# Key Features

- Behavioral Data Logging: Captures fine-grained keyboard and mouse metrics like typing intervals, cursor velocity, and click frequency.
- LSTM Model: A time-series model trained on 100-step sessions to classify fatigue state.
- Personalized & Hybrid Testing: Users can train their own models or use a hybrid of personal and global predictions.
- Session Labeling Workflow: User decides whether to label/save a session after recording.
- Flexible Recording Durations: Supports 2, 5, and 10-minute sessions for training/testing.

# Model Performance

- Model Type: LSTM-based sequential classifier  
- Evaluation Metrics: Precision, Recall, F1-score  
- Test Results:  
  - Fatigued Recall: 96%  
  - Relaxed Precision: 83%  
  - Overall Accuracy: 66%  

# How to Run Locally

> ✅ Requirements: Python 3.10, virtual environment recommended.

```bash
# Clone the repo
git clone https://github.com/your-username/cognitive-fatigue-detection.git
cd cognitive-fatigue-detection

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app/streamlit_app.py
