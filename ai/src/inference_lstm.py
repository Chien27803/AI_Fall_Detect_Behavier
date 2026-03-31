import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from datetime import datetime
import pygame
import os
import sys
import time
import requests

# ====== CONFIG ======
MODEL_PATH = "best_model.keras"
SCALER_PATH = "feature_scaler.npz"
NO_OF_TIMESTEPS = 35
NUM_FEATURES = 231
CLASS_NAMES = ["ADL", "BOXING", "FALL", "HAND_WAVING"]

CONFIDENCE_THRESHOLD = 0.70
FALL_THRESHOLD = 0.80

PRED_HISTORY_SIZE = 5
FALL_HISTORY_TRIGGER = 3

ALARM_FILE = "tieng-coi-canh-bao.mp3"

# ====== TELEGRAM CONFIG ======
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = "8639607585:AAG7_lj5qkPOE6jarwBZOADdtZjzkLJX7XQ"
TELEGRAM_CHAT_ID = "8697469060"
FALL_CONFIRM_SECONDS = 5.0
TELEGRAM_TIMEOUT = (3, 10)

label = "Warmup..."
confidence_text = ""

pred_history = deque(maxlen=PRED_HISTORY_SIZE)
fall_history = deque(maxlen=PRED_HISTORY_SIZE)
alarm_playing = False

# ====== FALL EVENT STATE ======
# Mỗi lần label chuyển sang FALL -> tạo 1 event mới
# Giữ FALL đủ 5 giây -> gửi đúng 1 lần
# Vẫn FALL thêm 10s, 20s -> không gửi thêm
# Chỉ khi thoát FALL rồi quay lại FALL -> event mới
fall_event_active = False
fall_event_start_time = None
fall_event_sent = False

# ====== PATHS ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(BASE_DIR, ALARM_FILE)
MODEL_FULL_PATH = os.path.join(BASE_DIR, MODEL_PATH)
SCALER_FULL_PATH = os.path.join(BASE_DIR, SCALER_PATH)

# ====== LOAD MODEL + SCALER ======
try:
    model = tf.keras.models.load_model(MODEL_FULL_PATH)

    if not os.path.exists(SCALER_FULL_PATH):
        print(f"Không tìm thấy file scaler: {SCALER_FULL_PATH}")
        sys.exit()

    scaler_data = np.load(SCALER_FULL_PATH)
    feature_mean = scaler_data["mean"].astype(np.float32)
    feature_std = scaler_data["std"].astype(np.float32)
except Exception as e:
    print(f"Lỗi load model/scaler: {e}")
    sys.exit()

# ====== INIT AUDIO ======
try:
    pygame.mixer.init()
    if not os.path.exists(ALARM_PATH):
        print(f"Không tìm thấy file âm thanh: {ALARM_PATH}")
        sys.exit()
    pygame.mixer.music.load(ALARM_PATH)
except Exception as e:
    print(f"Lỗi khởi tạo âm thanh: {e}")
    sys.exit()

# ====== CAMERA ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam")
    sys.exit()

# ====== MEDIAPIPE ======
mpPose = mp.solutions.pose
pose = mpPose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

lm_list = []
warmup_frames = 30
frame_count = 0
prev_landmarks = None


def extract_current_landmarks(results):
    coords = []
    for lm in results.pose_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(coords, dtype=np.float32)


def make_landmark_timestep(results, prev_landmarks=None):
    current_landmarks = extract_current_landmarks(results)

    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24

    hip_center_x = (current_landmarks[LEFT_HIP_IDX, 0] + current_landmarks[RIGHT_HIP_IDX, 0]) / 2.0
    hip_center_y = (current_landmarks[LEFT_HIP_IDX, 1] + current_landmarks[RIGHT_HIP_IDX, 1]) / 2.0
    hip_center_z = (current_landmarks[LEFT_HIP_IDX, 2] + current_landmarks[RIGHT_HIP_IDX, 2]) / 2.0

    current_landmarks[:, 0] -= hip_center_x
    current_landmarks[:, 1] -= hip_center_y
    current_landmarks[:, 2] -= hip_center_z

    frame_features = []

    for i, lm in enumerate(current_landmarks):
        x, y, z, visibility = lm

        if prev_landmarks is None:
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            prev_x, prev_y, prev_z, _ = prev_landmarks[i]
            vx = x - prev_x
            vy = y - prev_y
            vz = z - prev_z

        frame_features.extend([x, y, z, visibility, vx, vy, vz])

    return frame_features, current_landmarks


def draw_landmark_on_image(results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img


def draw_class_on_image(label_text, conf_text, img):
    font = cv2.FONT_HERSHEY_SIMPLEX

    if label_text == "FALL":
        action_color = (0, 0, 255)
    elif label_text == "BOXING":
        action_color = (255, 0, 0)
    elif label_text == "ADL":
        action_color = (0, 255, 0)
    elif label_text == "HAND_WAVING":
        action_color = (0, 165, 255)
    elif label_text == "Uncertain":
        action_color = (0, 255, 255)
    else:
        action_color = (255, 255, 255)

    cv2.putText(img, f"Action: {label_text}", (10, 30), font, 0.8, action_color, 2, cv2.LINE_AA)
    cv2.putText(img, f"Confidence: {conf_text}", (10, 65), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return img


def draw_datetime_on_image(img):
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (255, 255, 255)
    padding = 10

    (text_width, _), _ = cv2.getTextSize(current_time, font, font_scale, thickness)
    x = img.shape[1] - text_width - padding
    y = 30

    cv2.putText(
        img,
        current_time,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return img


def start_alarm():
    global alarm_playing
    if not alarm_playing:
        try:
            pygame.mixer.music.play(-1)
            alarm_playing = True
        except Exception as e:
            print(f"Không phát được âm thanh: {e}")


def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False


def should_trigger_fall_alarm():
    return sum(fall_history) >= FALL_HISTORY_TRIGGER


def handle_alarm():
    if should_trigger_fall_alarm():
        start_alarm()
    else:
        stop_alarm()


def send_telegram_photo(frame, caption):
    if not TELEGRAM_ENABLED:
        return False

    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "8639607585:AAG7_lj5qkPOE6jarwBZOADdtZjzkLJX7XQ":
        print("Chưa cấu hình TELEGRAM_BOT_TOKEN")
        return False

    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "8697469060":
        print("Chưa cấu hình TELEGRAM_CHAT_ID")
        return False

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print("Không encode được ảnh để gửi Telegram")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    files = {
        "photo": ("fall_alert.jpg", buffer.tobytes(), "image/jpeg")
    }

    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption
    }

    try:
        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=TELEGRAM_TIMEOUT
        )

        print("Status code:", response.status_code)
        print("Response text:", response.text)

        response.raise_for_status()

        payload = response.json()
        if not payload.get("ok", False):
            print(f"Telegram API trả về lỗi: {payload}")
            return False

        print("Đã gửi cảnh báo FALL lên Telegram")
        return True

    except requests.RequestException as e:
        print(f"Lỗi gửi Telegram: {e}")
        return False


def handle_telegram_fall_alert(frame_to_send):
    global fall_event_active, fall_event_start_time, fall_event_sent

    now = time.monotonic()

    # Khi label chuyển sang FALL => bắt đầu 1 event mới
    if label == "FALL":
        if not fall_event_active:
            fall_event_active = True
            fall_event_start_time = now
            fall_event_sent = False

        fall_duration = now - fall_event_start_time

        # Chỉ gửi 1 lần cho mỗi event FALL
        if fall_duration >= FALL_CONFIRM_SECONDS and not fall_event_sent:
            event_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            caption = (
                "⚠️ CẢNH BÁO TÉ NGÃ\n"
                f"Thời gian: {event_time}\n"
                f"Nhãn: {label}\n"
                f"Confidence: {confidence_text}\n"
                f"FALL liên tục: {fall_duration:.1f}s"
            )

            if send_telegram_photo(frame_to_send, caption):
                fall_event_sent = True
    else:
        # Khi label không còn là FALL => kết thúc event
        fall_event_active = False
        fall_event_start_time = None
        fall_event_sent = False


def detect(model, lm_list):
    global label, confidence_text

    lm_array = np.array(lm_list, dtype=np.float32)

    if lm_array.shape != (NO_OF_TIMESTEPS, NUM_FEATURES):
        label = "Invalid input"
        confidence_text = "0.00"
        return

    lm_array = (lm_array - feature_mean) / feature_std
    lm_array = np.expand_dims(lm_array, axis=0)

    preds = model.predict(lm_array, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    if class_id >= len(CLASS_NAMES):
        label = "Unknown"
        confidence_text = f"{confidence:.2f}"
        fall_history.append(0)
        return

    raw_label = CLASS_NAMES[class_id]

    if raw_label == "FALL":
        is_confident = confidence >= FALL_THRESHOLD
    else:
        is_confident = confidence >= CONFIDENCE_THRESHOLD

    if is_confident:
        pred_history.append(raw_label)
        stable_label = Counter(pred_history).most_common(1)[0][0]
        label = stable_label
    else:
        label = "Uncertain"

    if raw_label == "FALL" and confidence >= FALL_THRESHOLD:
        fall_history.append(1)
    else:
        fall_history.append(0)

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

        c_lm, prev_landmarks = make_landmark_timestep(results, prev_landmarks)
        lm_list.append(c_lm)

        if len(lm_list) > NO_OF_TIMESTEPS:
            lm_list.pop(0)

        if len(lm_list) == NO_OF_TIMESTEPS:
            detect(model, lm_list)
    else:
        pred_history.clear()
        fall_history.clear()
        lm_list.clear()
        prev_landmarks = None

        fall_event_active = False
        fall_event_start_time = None
        fall_event_sent = False

        if frame_count > warmup_frames:
            label = "No pose detected"
            confidence_text = "0.00"

    img = draw_class_on_image(label, confidence_text, img)
    img = draw_datetime_on_image(img)

    frame_to_send = img.copy()

    handle_alarm()
    handle_telegram_fall_alert(frame_to_send)

    cv2.imshow("LSTM Action Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stop_alarm()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()