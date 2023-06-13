import pandas as pd
import numpy as np
from fastai.tabular.all import *
import cv2
import mediapipe as mp
import csv
import os
import math

# Start Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

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

# Process images
input_folder = 'img'
image_names = os.listdir(input_folder)

data = []
for filename in image_names:
    image = cv2.imread(os.path.join(input_folder, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get landmarks
    results = pose.process(image)

    row = []
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks
        row = [filename] + row
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

    data.append(row)

# Create dataframe
header = ['image']
for landmark in landmarks:
    header += [f'{landmark.name.lower()}_x', f'{landmark.name.lower()}_y', f'{landmark.name.lower()}_z']
for (_, _, m) in midpoint_pairs:
    header += [f'{m}_x', f'{m}_y', f'{m}_z']

df = pd.DataFrame(data, columns=header)

# Calculate angles and add columns
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

# Save dataframe to a CSV file
df.to_csv('testtwo.csv', index=False)
