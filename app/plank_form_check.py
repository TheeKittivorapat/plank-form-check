import pandas as pd
import numpy as np
import streamlit as st
from fastai.tabular.all import *
import cv2
import mediapipe as mp
import csv
import os
import math
import time

# Start Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Define landmarks
landmarks = [mp_pose.PoseLandmark.NOSE]
midpoint_pairs = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, "shoulder"),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, "elbow"),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, "wrist"),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, "hip"),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, "knee"),
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE, "ankle"),
    (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL, "heel"),
    (mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, "foot_index")
]

# Process frames
def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get landmarks
        results = pose.process(frame)

        row = []
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks
            for landmark in landmarks:
                landmark_point = pose_landmarks.landmark[landmark]
                row += [landmark_point.x, landmark_point.y, landmark_point.z]

            # Calculate midpoints
            for (l1, l2, m) in midpoint_pairs:
                p1 = pose_landmarks.landmark[l1]
                p2 = pose_landmarks.landmark[l2]
                midpoint_x = (p1.x + p2.x) / 2
                midpoint_y = (p1.y + p2.y) / 2
                midpoint_z = (p1.z + p2.z) / 2
                row += [midpoint_x, midpoint_y, midpoint_z]

        frames.append(row)

    cap.release()
    cv2.destroyAllWindows()

    return frames

# Load model
learn = load_learner('model.pkl')

# Streamlit app

import subprocess
subprocess.call(['apt-get', 'update'])
subprocess.call(['apt-get', 'install', '-y', 'libgl1-mesa-glx'])

st.title("Plank Form Checker")
st.write("Upload a video of you planking from the side for a form check.")

uploaded_file = st.file_uploader("Choose a video", type=["mp4"])

if uploaded_file is not None:
    # Save video to temp file
    temp_file = "plank_video.mp4"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process frames
    frames = process_video(temp_file)

    if len(frames) == 0:
        st.write("No pose landmarks detected. Please try another file.")
    else:
        # Create dataframe
        header = []
        for landmark in landmarks:
            header += [f'{landmark.name.lower()}_x', f'{landmark.name.lower()}_y', f'{landmark.name.lower()}_z']
        for (_, _, m) in midpoint_pairs:
            header += [f'{m}_x', f'{m}_y', f'{m}_z']

        df = pd.DataFrame(frames, columns=header)

        # Calculate angles
        landmarks = ['nose', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'heel', 'foot_index']

        angle_pairs = []
        for i in range(len(landmarks)):
            for j in range(i + 1, len(landmarks)):
                pair1 = landmarks[i]
                pair2 = landmarks[j]
                name = pair1[0] + pair2[0]
                angle_pairs.append((pair1, pair2, name))

        for (l1, l2, a) in angle_pairs:
            x1, y1, z1 = np.array(df[f'{l1}_x']), np.array(df[f'{l1}_y']), np.array(df[f'{l1}_z'])
            x2, y2, z2 = np.array(df[f'{l2}_x']), np.array(df[f'{l2}_y']), np.array(df[f'{l2}_z'])
            dot_product = x1 * x2 + y1 * y2 + z1 * z2
            norm1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
            norm2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
            cosine_angle = dot_product / (norm1 * norm2)
            angle = np.degrees(np.arccos(cosine_angle))
            df[f'{a}'] = angle

        # Remove old columns
        df.drop(df.filter(regex='_x|_y|_z').columns, axis=1, inplace=True)

        # Make prediction
        test_dl = learn.dls.test_dl(df)
        preds, _ = learn.get_preds(dl=test_dl)
        predicted_labels = preds.argmax(dim=1).numpy()

        label_dict = {0: 'Correct Form', 1: 'Hip is too high', 2: 'Hip is too low'}

        # Open video
        cap = cv2.VideoCapture(temp_file)

        video_placeholder = st.empty()

        label_placeholder = st.empty()

        # Loop through frames and show label
        for i, frame in enumerate(frames):
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            video_placeholder.image(img, channels="RGB", use_column_width=True)

            label_placeholder.text("Prediction: " + label_dict[predicted_labels[i]])

            time.sleep(0.1)

        cap.release()
        os.remove(temp_file)
