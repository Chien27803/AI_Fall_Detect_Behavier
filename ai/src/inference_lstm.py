import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# ====== CONFIG ======
MODEL_PATH = "best_model.keras"
NO_OF_TIMESTEPS = 30
NUM_FEATURES = 132
CLASS_NAMES = ["ADL","BOXING","FALL"]
CONFIDENCE_THRESHOLD = 0.7

label = "Warmup..."
confidence_text = ""

pred_history = deque(maxlen=5)

# ====== LOAD MODEL ======
model = tf.keras.models.load_model(MODEL_PATH)

# ====== CAMERA ======
cap = cv2.VideoCapture(0)

# ====== MEDIAPIPE ======
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
warmup_frames = 30
frame_count = 0


def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm


def draw_landmark_on_image(results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img


def draw_class_on_image(label_text, conf_text, img):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, f"Action: {label_text}", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Confidence: {conf_text}", (10, 65), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Press 'q' to quit", (10, 100), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def detect(model, lm_list):
    global label, confidence_text

    lm_array = np.array(lm_list, dtype=np.float32)

    if lm_array.shape != (NO_OF_TIMESTEPS, NUM_FEATURES):
        label = "Invalid input"
        confidence_text = "0.00"
        return

    lm_array = np.expand_dims(lm_array, axis=0)

    preds = model.predict(lm_array, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    if class_id >= len(CLASS_NAMES):
        label = "Unknown"
        confidence_text = f"{confidence:.2f}"
        return

    if confidence < CONFIDENCE_THRESHOLD:
        current_label = "Uncertain"
    else:
        current_label = CLASS_NAMES[class_id]

    pred_history.append(current_label)
    smoothed_label = max(set(pred_history), key=pred_history.count)

    label = smoothed_label
    confidence_text = f"{confidence:.2f}"


while True:
    success, img = cap.read()
    if not success:
        continue

    frame_count += 1

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if frame_count > warmup_frames and results.pose_landmarks:
        img = draw_landmark_on_image(results, img)

        c_lm = make_landmark_timestep(results)
        lm_list.append(c_lm)

        if len(lm_list) > NO_OF_TIMESTEPS:
            lm_list.pop(0)

        if len(lm_list) == NO_OF_TIMESTEPS:
            detect(model, lm_list)
    else:
        pred_history.clear()
        lm_list.clear()
        if frame_count > warmup_frames:
            label = "No pose detected"
            confidence_text = "0.00"

    img = draw_class_on_image(label, confidence_text, img)

    cv2.imshow("LSTM Action Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()