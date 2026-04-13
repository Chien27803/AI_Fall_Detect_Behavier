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

# ====== FALL ENTRY FILTER CONFIG ======
# Model đoán FALL vẫn chưa đủ.
# Chỉ khi hình học landmark cho thấy người đang ở tư thế nằm / đổ ngang rõ ràng
# thì mới xem xét cho vào trạng thái FALL.
FALL_POSTURE_VISIBILITY_THRESHOLD = 0.50
LYING_TORSO_ANGLE_MAX = 35.0
LYING_BODY_AXIS_ANGLE_MAX = 40.0
LYING_SHOULDER_HIP_Y_MAX = 0.10
LYING_HIP_KNEE_Y_MAX = 0.12
LYING_KNEE_ANKLE_Y_MAX = 0.14
LYING_HORIZONTAL_RATIO_MIN = 1.20

# ====== RECENT FALL TRANSITION CONFIG ======
# Điều kiện vào FALL mới:
# - model đoán FALL
# - trong khoảng 1-2 giây gần đây, vai có chuyển từ cao xuống thấp rõ rệt
# Lưu ý: trên ảnh, trục y tăng nghĩa là điểm bị hạ thấp xuống.
SHOULDER_HISTORY_MAXLEN = 120
RECENT_SHOULDER_DROP_SECONDS = 1.5
SHOULDER_DROP_Y_THRESHOLD = 0.16
MIN_SHOULDER_VISIBILITY = 0.50

# ====== FALL HOLD / RECOVERY CONFIG ======
# Khi đã vào FALL thì CHỈ thoát khi người đó đứng thẳng dậy rõ ràng.
# Ngồi dậy, chống tay dậy, quỳ, nửa ngồi nửa đứng... vẫn tiếp tục giữ FALL.
RECOVERY_CONSEC_FRAMES = 10
RECOVERY_VISIBILITY_THRESHOLD = 0.50
TORSO_UPRIGHT_ANGLE_MIN = 50.0
HEAD_HIP_Y_MARGIN = 0.03
SHOULDER_HIP_Y_MARGIN = 0.02
STAND_HIP_KNEE_MARGIN = 0.08
STAND_KNEE_ANKLE_MARGIN = 0.04

# ====== TELEGRAM CONFIG ======
TELEGRAM_ENABLED = False  # True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TIMEOUT = (3, 10)
FALL_CONFIRM_SECONDS = 5.0
FALL_END_GRACE_SECONDS = 1.5

label = "Warmup..."
model_label = "Warmup..."
confidence_text = ""

# test .env
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

# ====== FALL HOLD STATE ======
fall_latched = False
recovery_frame_count = 0

# ====== RECENT MOTION CONTEXT ======
shoulder_y_history = deque(maxlen=SHOULDER_HISTORY_MAXLEN)

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


def get_fall_entry_posture(raw_landmarks: np.ndarray) -> str:
    """
    Chỉ dùng để xác nhận có cho phép BẮT ĐẦU FALL hay không.

    Trả về:
    - 'lying'            : người đang nằm / đổ ngang đủ rõ để xét vào FALL
    - 'not_lying'        : chưa đủ dấu hiệu nằm
    - 'unknown'          : landmark kém tin cậy, chưa kết luận
    """
    if raw_landmarks is None or raw_landmarks.shape != (33, 4):
        return "unknown"

    important_idxs = [0, 11, 12, 23, 24, 25, 26, 27, 28]
    important_visibility = float(np.mean(raw_landmarks[important_idxs, 3]))
    if important_visibility < FALL_POSTURE_VISIBILITY_THRESHOLD:
        return "unknown"

    nose = raw_landmarks[0, :3]
    left_shoulder = raw_landmarks[11, :3]
    right_shoulder = raw_landmarks[12, :3]
    left_hip = raw_landmarks[23, :3]
    right_hip = raw_landmarks[24, :3]
    left_knee = raw_landmarks[25, :3]
    right_knee = raw_landmarks[26, :3]
    left_ankle = raw_landmarks[27, :3]
    right_ankle = raw_landmarks[28, :3]

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    knee_center = (left_knee + right_knee) / 2.0
    ankle_center = (left_ankle + right_ankle) / 2.0

    torso_dx = float(shoulder_center[0] - hip_center[0])
    torso_dy = float(shoulder_center[1] - hip_center[1])
    torso_angle_deg = float(np.degrees(np.arctan2(abs(torso_dy), abs(torso_dx) + 1e-6)))

    body_dx = float(ankle_center[0] - shoulder_center[0])
    body_dy = float(ankle_center[1] - shoulder_center[1])
    body_axis_angle_deg = float(np.degrees(np.arctan2(abs(body_dy), abs(body_dx) + 1e-6)))

    shoulder_hip_same_level = abs(float(shoulder_center[1] - hip_center[1])) <= LYING_SHOULDER_HIP_Y_MAX
    hip_knee_same_level = abs(float(hip_center[1] - knee_center[1])) <= LYING_HIP_KNEE_Y_MAX
    knee_ankle_same_level = abs(float(knee_center[1] - ankle_center[1])) <= LYING_KNEE_ANKLE_Y_MAX

    body_points = np.array([
        nose[:2],
        shoulder_center[:2],
        hip_center[:2],
        knee_center[:2],
        ankle_center[:2],
    ], dtype=np.float32)

    vertical_span = float(np.max(body_points[:, 1]) - np.min(body_points[:, 1]))
    horizontal_span = float(np.max(body_points[:, 0]) - np.min(body_points[:, 0]))
    horizontal_dominant = horizontal_span >= (vertical_span * LYING_HORIZONTAL_RATIO_MIN)

    lying = (
        torso_angle_deg <= LYING_TORSO_ANGLE_MAX
        and body_axis_angle_deg <= LYING_BODY_AXIS_ANGLE_MAX
        and shoulder_hip_same_level
        and hip_knee_same_level
        and knee_ankle_same_level
        and horizontal_dominant
    )

    if lying:
        return "lying"
    return "not_lying"


def get_recovery_posture(raw_landmarks: np.ndarray) -> str:
    """
    Chỉ dùng để quyết định khi nào được thoát khỏi trạng thái FALL.

    Nguyên tắc:
    - CHỈ khi người dùng đứng thẳng rõ ràng mới được coi là recovered.
    - Nếu mới ngồi dậy, quỳ, chống tay, nửa ngồi nửa đứng... => vẫn là FALL.

    Trả về một trong các giá trị:
    - 'standing'
    - 'not_recovered'
    - 'unknown'
    """
    if raw_landmarks is None or raw_landmarks.shape != (33, 4):
        return "unknown"

    important_idxs = [0, 11, 12, 23, 24, 25, 26, 27, 28]
    important_visibility = float(np.mean(raw_landmarks[important_idxs, 3]))
    if important_visibility < RECOVERY_VISIBILITY_THRESHOLD:
        return "unknown"

    nose = raw_landmarks[0, :3]
    left_shoulder = raw_landmarks[11, :3]
    right_shoulder = raw_landmarks[12, :3]
    left_hip = raw_landmarks[23, :3]
    right_hip = raw_landmarks[24, :3]
    left_knee = raw_landmarks[25, :3]
    right_knee = raw_landmarks[26, :3]
    left_ankle = raw_landmarks[27, :3]
    right_ankle = raw_landmarks[28, :3]

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    knee_center = (left_knee + right_knee) / 2.0
    ankle_center = (left_ankle + right_ankle) / 2.0

    torso_dx = float(shoulder_center[0] - hip_center[0])
    torso_dy = float(shoulder_center[1] - hip_center[1])
    torso_angle_deg = float(np.degrees(np.arctan2(abs(torso_dy), abs(torso_dx) + 1e-6)))

    # Trong ảnh: y càng nhỏ thì điểm càng ở cao hơn
    head_above_hip = nose[1] < (hip_center[1] - HEAD_HIP_Y_MARGIN)
    shoulders_above_hip = shoulder_center[1] < (hip_center[1] - SHOULDER_HIP_Y_MARGIN)
    torso_upright = torso_angle_deg >= TORSO_UPRIGHT_ANGLE_MIN

    hips_above_knees = hip_center[1] < (knee_center[1] - STAND_HIP_KNEE_MARGIN)
    knees_above_ankles = knee_center[1] < (ankle_center[1] - STAND_KNEE_ANKLE_MARGIN)

    standing = (
        torso_upright
        and head_above_hip
        and shoulders_above_hip
        and hips_above_knees
        and knees_above_ankles
    )

    if standing:
        return "standing"
    return "not_recovered"


def reset_motion_context():
    shoulder_y_history.clear()


def update_motion_context(raw_landmarks: np.ndarray):
    """
    Lưu lịch sử độ cao vai gần đây để kiểm tra xem vai có vừa bị hạ thấp xuống không.
    """
    if raw_landmarks is None or raw_landmarks.shape != (33, 4):
        return

    left_shoulder_vis = float(raw_landmarks[11, 3])
    right_shoulder_vis = float(raw_landmarks[12, 3])
    shoulder_visibility_mean = (left_shoulder_vis + right_shoulder_vis) / 2.0

    if shoulder_visibility_mean < MIN_SHOULDER_VISIBILITY:
        return

    shoulder_center_y = float((raw_landmarks[11, 1] + raw_landmarks[12, 1]) / 2.0)
    now = time.monotonic()
    shoulder_y_history.append((now, shoulder_center_y))

    max_age = RECENT_SHOULDER_DROP_SECONDS + 0.5
    while shoulder_y_history and (now - shoulder_y_history[0][0]) > max_age:
        shoulder_y_history.popleft()


def get_recent_shoulder_drop_signal():
    """
    Trả về:
    - recent_shoulder_drop: vai có vừa bị hạ từ cao xuống thấp trong 1-2 giây gần đây không
    - shoulder_drop_delta: độ chênh y của vai trong cửa sổ thời gian gần đây

    Trên ảnh:
    - y nhỏ hơn = vai cao hơn
    - y lớn hơn = vai thấp hơn
    """
    if not shoulder_y_history:
        return False, 0.0

    now = time.monotonic()
    recent_points = [y for ts, y in shoulder_y_history if (now - ts) <= RECENT_SHOULDER_DROP_SECONDS]

    if len(recent_points) < 3:
        return False, 0.0

    current_shoulder_y = recent_points[-1]
    previous_highest_shoulder_y = min(recent_points[:-1]) if len(recent_points) > 1 else current_shoulder_y
    shoulder_drop_delta = float(current_shoulder_y - previous_highest_shoulder_y)
    recent_shoulder_drop = shoulder_drop_delta >= SHOULDER_DROP_Y_THRESHOLD

    return recent_shoulder_drop, shoulder_drop_delta


def apply_fall_hold(current_model_label: str, raw_landmarks: np.ndarray) -> str:
    """
    Chỉ can thiệp riêng cho FALL:
    - Nếu model đoán FALL thì phải có thêm dấu hiệu vai vừa bị hạ thấp xuống rõ rệt trong 1-2 giây gần đây
    - Nếu không có shoulder-drop gần đây thì không cho vào FALL
    - Nếu đã vào FALL => GIỮ FALL cho đến khi người dùng đứng thẳng dậy liên tiếp
    - Ngồi dậy KHÔNG được thoát FALL
    """
    global fall_latched, recovery_frame_count

    if not fall_latched:
        if current_model_label == "FALL":
            recent_shoulder_drop, _ = get_recent_shoulder_drop_signal()

            if recent_shoulder_drop:
                fall_latched = True
                recovery_frame_count = 0
                return "FALL"

            return "ADL"

        return current_model_label

    posture = get_recovery_posture(raw_landmarks)

    # CHỈ chấp nhận standing là recovered
    if posture == "standing":
        recovery_frame_count += 1
    else:
        recovery_frame_count = 0
        return "FALL"

    if recovery_frame_count >= RECOVERY_CONSEC_FRAMES:
        fall_latched = False
        recovery_frame_count = 0
        pred_history.clear()   # giảm nguy cơ model bị trễ vài frame rồi vào FALL lại ngay
        reset_motion_context() # reset ngữ cảnh cũ sau khi đã đứng dậy lại
        return "ADL"

    return "FALL"


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
    global model_label, confidence_text

    lm_array = np.array(lm_list, dtype=np.float32)

    if lm_array.shape != (NO_OF_TIMESTEPS, NUM_FEATURES):
        model_label = "Invalid input"
        confidence_text = "0.00"
        return

    lm_array = np.expand_dims(lm_array, axis=0)

    preds = model.predict(lm_array, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    if class_id >= len(CLASS_NAMES):
        model_label = "Unknown"
        confidence_text = f"{confidence:.2f}"
        return

    if confidence < CONFIDENCE_THRESHOLD:
        current_label = "Uncertain"
    else:
        current_label = CLASS_NAMES[class_id]

    pred_history.append(current_label)
    smoothed_label = Counter(pred_history).most_common(1)[0][0]

    model_label = smoothed_label
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

            raw_landmarks = extract_current_landmarks(results)
            update_motion_context(raw_landmarks)

            c_lm, new_prev_landmarks = make_landmark_timestep(results, prev_landmarks)

            # chỉ thêm vào sequence nếu frame đủ chất lượng
            if c_lm is not None:
                prev_landmarks = new_prev_landmarks
                lm_list.append(c_lm)

                if len(lm_list) > NO_OF_TIMESTEPS:
                    lm_list.pop(0)

                if len(lm_list) == NO_OF_TIMESTEPS and frame_count % PREDICT_EVERY_N_FRAMES == 0:
                    detect(model, lm_list)

                label = apply_fall_hold(model_label, raw_landmarks)
            else:
                if fall_latched:
                    label = "FALL"
                else:
                    label = "Low visibility"
                    confidence_text = "0.00"

        else:
            pred_history.clear()
            lm_list.clear()
            prev_landmarks = None

            if not fall_latched:
                reset_motion_context()

            if frame_count > warmup_frames:
                if fall_latched:
                    label = "FALL"
                else:
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
