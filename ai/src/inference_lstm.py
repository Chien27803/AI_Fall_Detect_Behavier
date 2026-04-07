import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import requests
import time
import threading
from collections import deque, Counter
from datetime import datetime
import pygame
import os
import sys
from dotenv import load_dotenv

# ====== BASE DIR ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== LOAD BIẾN MÔI TRƯỜNG ======
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ====== CONFIG ======
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
NO_OF_TIMESTEPS = 35
NUM_FEATURES = 231
CLASS_NAMES = ["ADL", "BOXING", "FALL", "HAND_WAVING"]

# giảm nhẹ để bắt FALL nhạy hơn
CONFIDENCE_THRESHOLD = 0.75

ALARM_FILE = "tieng-coi-canh-bao.mp3"
ALARM_PATH = os.path.join(BASE_DIR, ALARM_FILE)

# Chỉ predict mỗi 2 frame
PREDICT_EVERY_N_FRAMES = 2

# ====== VISIBILITY FILTER ======
# Nếu visibility trung bình của 33 landmark thấp hơn ngưỡng này
# thì bỏ qua frame đó, không đưa vào lm_list
MIN_VISIBILITY_MEAN = 0.35

# ====== TELEGRAM CONFIG ======
TELEGRAM_ENABLED = False  # True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TIMEOUT = (3, 10)
FALL_CONFIRM_SECONDS = 5.0
FALL_END_GRACE_SECONDS = 1.5

label = "Warmup..."
confidence_text = ""

#test .env
print("BOT:", TELEGRAM_BOT_TOKEN)
print("CHAT_ID:", TELEGRAM_CHAT_ID)

pred_history = deque(maxlen=5)
alarm_playing = False

# ====== FPS CONFIG ======
fps_history = deque(maxlen=30)
runtime_start_time = time.perf_counter()
prev_frame_time = None
display_fps = 0.0
avg_fps = 0.0
processed_frame_count = 0

# ====== FALL EVENT STATE FOR TELEGRAM ======
fall_event_active = False
fall_event_start_time = None
fall_event_sent = False
fall_last_fall_time = None

# ====== LOAD MODEL ======
if not os.path.exists(MODEL_PATH):
    print(f"Không tìm thấy model: {MODEL_PATH}")
    sys.exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Lỗi load model: {e}")
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
    return np.array(coords, dtype=np.float32)  # shape (33, 4)


def is_frame_visibility_valid(current_landmarks: np.ndarray) -> bool:
    """
    current_landmarks shape: (33, 4)
    cột thứ 4 là visibility
    """
    if current_landmarks is None or current_landmarks.shape != (33, 4):
        return False

    visibility_mean = float(np.mean(current_landmarks[:, 3]))
    return visibility_mean >= MIN_VISIBILITY_MEAN


def make_landmark_timestep(results, prev_landmarks=None):
    current_landmarks = extract_current_landmarks(results)  # (33, 4)

    # ===== LỌC FRAME THEO VISIBILITY =====
    if not is_frame_visibility_valid(current_landmarks):
        return None, prev_landmarks

    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24

    hip_center_x = (
        current_landmarks[LEFT_HIP_IDX, 0] + current_landmarks[RIGHT_HIP_IDX, 0]
    ) / 2.0
    hip_center_y = (
        current_landmarks[LEFT_HIP_IDX, 1] + current_landmarks[RIGHT_HIP_IDX, 1]
    ) / 2.0
    hip_center_z = (
        current_landmarks[LEFT_HIP_IDX, 2] + current_landmarks[RIGHT_HIP_IDX, 2]
    ) / 2.0

    # Chuẩn hóa x, y, z theo tâm hông
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
    elif label_text == "Low visibility":
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

    (text_width, text_height), _ = cv2.getTextSize(
        current_time, font, font_scale, thickness
    )

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


def draw_fps_on_image(img, fps, avg_fps):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        img,
        f"FPS: {fps:.1f}",
        (10, img.shape[0] - 40),
        font,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        f"AVG FPS: {avg_fps:.1f}",
        (10, img.shape[0] - 10),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return img


def update_fps():
    global prev_frame_time, display_fps, avg_fps, processed_frame_count

    current_time = time.perf_counter()
    processed_frame_count += 1

    if prev_frame_time is not None:
        delta = current_time - prev_frame_time
        if delta > 0:
            instant_fps = 1.0 / delta
            fps_history.append(instant_fps)
            display_fps = sum(fps_history) / len(fps_history)

    prev_frame_time = current_time

    elapsed = current_time - runtime_start_time
    if elapsed > 0:
        avg_fps = processed_frame_count / elapsed


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


def handle_alarm(current_label):
    if current_label == "FALL":
        start_alarm()
    else:
        stop_alarm()


def send_telegram_photo(frame, caption):
    if not TELEGRAM_ENABLED:
        return False

    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
        print("Chưa cấu hình TELEGRAM_BOT_TOKEN")
        return False

    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID":
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
        print("Telegram status:", response.status_code)
        print("Telegram response:", response.text)
        response.raise_for_status()

        payload = response.json()
        if not payload.get("ok", False):
            print(f"Telegram API trả về lỗi: {payload}")
            return False

        print("Đã gửi ảnh cảnh báo FALL lên Telegram")
        return True
    except requests.RequestException as e:
        print(f"Lỗi gửi Telegram: {e}")
        return False


def send_telegram_photo_async(frame, caption):
    frame_copy = frame.copy()

    def worker():
        send_telegram_photo(frame_copy, caption)

    threading.Thread(target=worker, daemon=True).start()
    return True


def handle_telegram_fall_alert(current_label, frame):
    global fall_event_active, fall_event_start_time, fall_event_sent, fall_last_fall_time

    now = time.monotonic()

    if current_label == "FALL":
        fall_last_fall_time = now

        if not fall_event_active:
            fall_event_active = True
            fall_event_start_time = now
            fall_event_sent = False
            print("Bắt đầu sự kiện FALL, đang đếm 5 giây...")

        fall_duration = now - fall_event_start_time

        if not fall_event_sent and fall_duration >= FALL_CONFIRM_SECONDS:
            caption = (
                "⚠️ CẢNH BÁO TÉ NGÃ\n"
                f"Thời gian: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"Nhãn: {current_label}\n"
                f"Confidence: {confidence_text}\n"
                f"FALL liên tục: {fall_duration:.1f}s"
            )

            if send_telegram_photo_async(frame, caption):
                fall_event_sent = True
                print("Đã kích hoạt gửi ảnh Telegram ở luồng nền")
    else:
        if fall_event_active and fall_last_fall_time is not None:
            time_since_last_fall = now - fall_last_fall_time
            if time_since_last_fall >= FALL_END_GRACE_SECONDS:
                print("Sự kiện FALL đã kết thúc, reset trạng thái Telegram")
                fall_event_active = False
                fall_event_start_time = None
                fall_event_sent = False
                fall_last_fall_time = None


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
    smoothed_label = Counter(pred_history).most_common(1)[0][0]

    label = smoothed_label
    confidence_text = f"{confidence:.2f}"


try:
    while True:
        success, img = cap.read()
        if not success:
            continue

        update_fps()
        frame_count += 1

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if frame_count > warmup_frames and results.pose_landmarks:
            img = draw_landmark_on_image(results, img)

            c_lm, new_prev_landmarks = make_landmark_timestep(results, prev_landmarks)

            # chỉ thêm vào sequence nếu frame đủ chất lượng
            if c_lm is not None:
                prev_landmarks = new_prev_landmarks
                lm_list.append(c_lm)

                if len(lm_list) > NO_OF_TIMESTEPS:
                    lm_list.pop(0)

                if len(lm_list) == NO_OF_TIMESTEPS and frame_count % PREDICT_EVERY_N_FRAMES == 0:
                    detect(model, lm_list)
            else:
                if frame_count > warmup_frames:
                    label = "Low visibility"
                    confidence_text = "0.00"

        else:
            pred_history.clear()
            lm_list.clear()
            prev_landmarks = None
            if frame_count > warmup_frames:
                label = "No pose detected"
                confidence_text = "0.00"

        img = draw_class_on_image(label, confidence_text, img)
        img = draw_datetime_on_image(img)
        img = draw_fps_on_image(img, display_fps, avg_fps)

        handle_alarm(label)
        handle_telegram_fall_alert(label, img)

        cv2.imshow("LSTM Action Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()