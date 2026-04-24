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
# - trong 1.5 giây gần nhất, 1 hoặc 2 vai có bị hạ từ cao xuống thấp
#   VÀ vị trí hiện tại của vai đó đang sát mặt đất
#   HOẶC mông/hông đang chạm sát mặt đất VÀ 2 chân duỗi tương đối thẳng.
#
# Quy ước MediaPipe/camera:
# - y nhỏ hơn = điểm ở cao hơn
# - y lớn hơn = điểm bị hạ thấp xuống
# - mặt đất tạm tính bằng vị trí bàn chân/cổ chân thấp nhất nhìn thấy được.
SHOULDER_HISTORY_MAXLEN = 120
RECENT_SHOULDER_DROP_SECONDS = 1.5
SHOULDER_DROP_Y_THRESHOLD = 0.16
MIN_SHOULDER_VISIBILITY = 0.50

SHOULDER_NEAR_GROUND_MARGIN = 0.15
HIP_NEAR_GROUND_MARGIN = 0.12
MIN_HIP_VISIBILITY = 0.50
MIN_KNEE_VISIBILITY = 0.50
MIN_ANKLE_VISIBILITY_FOR_GROUND = 0.35

# Điều kiện cho nhánh mông/hông chạm đất:
# 2 chân phải duỗi tương đối thẳng thì mới cho phép vào FALL.
# Góc gối càng gần 180 độ nghĩa là chân càng thẳng.
LEG_STRAIGHT_KNEE_ANGLE_MIN = 145.0

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
left_shoulder_y_history = deque(maxlen=SHOULDER_HISTORY_MAXLEN)
right_shoulder_y_history = deque(maxlen=SHOULDER_HISTORY_MAXLEN)

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
    left_shoulder_y_history.clear()
    right_shoulder_y_history.clear()


def get_ground_y_from_feet(raw_landmarks: np.ndarray):
    """
    Lấy vị trí mặt đất tương đối theo bàn chân/cổ chân.
    Trong ảnh, y càng lớn thì càng thấp. Vì vậy ground_y lấy theo điểm chân có y lớn nhất.
    Ưu tiên ankle, nếu ankle visibility kém thì dùng thêm heel/foot_index.
    """
    if raw_landmarks is None or raw_landmarks.shape != (33, 4):
        return None

    # 27,28: ankle | 29,30: heel | 31,32: foot_index
    foot_indices = [27, 28, 29, 30, 31, 32]
    visible_foot_y = [
        float(raw_landmarks[idx, 1])
        for idx in foot_indices
        if float(raw_landmarks[idx, 3]) >= MIN_ANKLE_VISIBILITY_FOR_GROUND
    ]

    if not visible_foot_y:
        return None

    return max(visible_foot_y)


def update_motion_context(raw_landmarks: np.ndarray):
    """
    Lưu lịch sử độ cao vai trái/phải gần đây để kiểm tra:
    - trong 1.5s gần nhất vai có bị hạ từ cao xuống thấp không.
    """
    if raw_landmarks is None or raw_landmarks.shape != (33, 4):
        return

    now = time.monotonic()

    left_shoulder_vis = float(raw_landmarks[11, 3])
    right_shoulder_vis = float(raw_landmarks[12, 3])

    if left_shoulder_vis >= MIN_SHOULDER_VISIBILITY:
        left_shoulder_y_history.append((now, float(raw_landmarks[11, 1])))

    if right_shoulder_vis >= MIN_SHOULDER_VISIBILITY:
        right_shoulder_y_history.append((now, float(raw_landmarks[12, 1])))

    max_age = RECENT_SHOULDER_DROP_SECONDS + 0.5

    while left_shoulder_y_history and (now - left_shoulder_y_history[0][0]) > max_age:
        left_shoulder_y_history.popleft()

    while right_shoulder_y_history and (now - right_shoulder_y_history[0][0]) > max_age:
        right_shoulder_y_history.popleft()


def get_recent_single_shoulder_drop_signal(shoulder_history: deque, current_shoulder_y: float):
    """
    Kiểm tra 1 vai có vừa bị hạ từ cao xuống thấp trong 1.5 giây gần đây không.
    """
    if not shoulder_history:
        return False, 0.0

    now = time.monotonic()
    recent_points = [y for ts, y in shoulder_history if (now - ts) <= RECENT_SHOULDER_DROP_SECONDS]

    if len(recent_points) < 3:
        return False, 0.0

    previous_highest_shoulder_y = min(recent_points[:-1])
    shoulder_drop_delta = float(current_shoulder_y - previous_highest_shoulder_y)
    recent_shoulder_drop = shoulder_drop_delta >= SHOULDER_DROP_Y_THRESHOLD

    return recent_shoulder_drop, shoulder_drop_delta


def is_shoulder_near_ground(current_shoulder_y: float, ground_y: float) -> bool:
    return current_shoulder_y >= (ground_y - SHOULDER_NEAR_GROUND_MARGIN)


def get_recent_shoulder_drop_near_ground_signal(raw_landmarks: np.ndarray):
    """
    True nếu trong 1.5s gần đây có 1 hoặc 2 vai bị hạ từ cao xuống thấp
    và hiện tại vai đó đang sát mặt đất.
    """
    if raw_landmarks is None or raw_landmarks.shape != (33, 4):
        return False, "invalid_landmarks"

    ground_y = get_ground_y_from_feet(raw_landmarks)
    if ground_y is None:
        return False, "ground_unknown"

    left_signal = False
    right_signal = False
    left_delta = 0.0
    right_delta = 0.0

    left_vis = float(raw_landmarks[11, 3])
    if left_vis >= MIN_SHOULDER_VISIBILITY:
        left_y = float(raw_landmarks[11, 1])
        left_drop, left_delta = get_recent_single_shoulder_drop_signal(left_shoulder_y_history, left_y)
        left_signal = left_drop and is_shoulder_near_ground(left_y, ground_y)

    right_vis = float(raw_landmarks[12, 3])
    if right_vis >= MIN_SHOULDER_VISIBILITY:
        right_y = float(raw_landmarks[12, 1])
        right_drop, right_delta = get_recent_single_shoulder_drop_signal(right_shoulder_y_history, right_y)
        right_signal = right_drop and is_shoulder_near_ground(right_y, ground_y)

    detail = (
        f"left={left_signal}({left_delta:.3f}), "
        f"right={right_signal}({right_delta:.3f}), "
        f"ground_y={ground_y:.3f}"
    )

    return left_signal or right_signal, detail


def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Tính góc ABC theo mặt phẳng ảnh 2D.
    Dùng cho góc gối: hip - knee - ankle.
    Góc gần 180 độ nghĩa là chân tương đối thẳng.
    """
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def are_both_legs_relatively_straight(raw_landmarks: np.ndarray):
    """
    Kiểm tra 2 chân có duỗi tương đối thẳng không.
    Dùng góc tại đầu gối:
    - chân trái: left_hip - left_knee - left_ankle
    - chân phải: right_hip - right_knee - right_ankle
    """
    if raw_landmarks is None or raw_landmarks.shape != (33, 4):
        return False, "invalid_landmarks"

    left_idxs = [23, 25, 27]
    right_idxs = [24, 26, 28]

    left_visibility = float(np.mean(raw_landmarks[left_idxs, 3]))
    right_visibility = float(np.mean(raw_landmarks[right_idxs, 3]))

    min_leg_visibility = min(MIN_HIP_VISIBILITY, MIN_KNEE_VISIBILITY, MIN_ANKLE_VISIBILITY_FOR_GROUND)

    if left_visibility < min_leg_visibility or right_visibility < min_leg_visibility:
        return False, f"leg_low_visibility left={left_visibility:.2f}, right={right_visibility:.2f}"

    left_angle = calculate_angle_2d(raw_landmarks[23], raw_landmarks[25], raw_landmarks[27])
    right_angle = calculate_angle_2d(raw_landmarks[24], raw_landmarks[26], raw_landmarks[28])

    left_straight = left_angle >= LEG_STRAIGHT_KNEE_ANGLE_MIN
    right_straight = right_angle >= LEG_STRAIGHT_KNEE_ANGLE_MIN
    both_straight = left_straight and right_straight

    detail = (
        f"legs_straight={both_straight}, "
        f"left_knee_angle={left_angle:.1f}, right_knee_angle={right_angle:.1f}"
    )
    return both_straight, detail


def get_hip_touch_ground_with_straight_legs_signal(raw_landmarks: np.ndarray):
    """
    Kiểm tra nhánh mông/hông:
    - mông/hông chạm sát mặt đất
    - VÀ 2 chân phải duỗi tương đối thẳng

    Vì MediaPipe không có landmark riêng cho mông, dùng tâm 2 hông trái/phải làm xấp xỉ.
    """
    if raw_landmarks is None or raw_landmarks.shape != (33, 4):
        return False, "invalid_landmarks"

    ground_y = get_ground_y_from_feet(raw_landmarks)
    if ground_y is None:
        return False, "ground_unknown"

    left_hip_vis = float(raw_landmarks[23, 3])
    right_hip_vis = float(raw_landmarks[24, 3])
    hip_visibility_mean = (left_hip_vis + right_hip_vis) / 2.0

    if hip_visibility_mean < MIN_HIP_VISIBILITY:
        return False, "hip_low_visibility"

    hip_center_y = float((raw_landmarks[23, 1] + raw_landmarks[24, 1]) / 2.0)
    hip_near_ground = hip_center_y >= (ground_y - HIP_NEAR_GROUND_MARGIN)

    legs_straight, legs_detail = are_both_legs_relatively_straight(raw_landmarks)
    hip_ok = hip_near_ground and legs_straight

    detail = (
        f"hip_near_ground={hip_near_ground}, hip_y={hip_center_y:.3f}, "
        f"ground_y={ground_y:.3f}, {legs_detail}"
    )
    return hip_ok, detail


def apply_fall_hold(current_model_label: str, raw_landmarks: np.ndarray) -> str:
    """
    Chỉ can thiệp riêng cho FALL:
    - Nếu model đoán FALL thì phải check thêm:
        + trong 1.5s gần nhất, 1 hoặc 2 vai bị hạ từ cao xuống thấp
          và vai đó đang sát mặt đất, mặt đất tính theo bàn chân/cổ chân
          HOẶC
        + mông/hông đang chạm sát mặt đất VÀ 2 chân duỗi tương đối thẳng
    - Nếu không có các tín hiệu trên thì không cho label_text hiển thị FALL
    - Nếu đã vào FALL => GIỮ FALL cho đến khi người dùng đứng thẳng dậy liên tiếp
    - Ngồi dậy KHÔNG được thoát FALL
    """
    global fall_latched, recovery_frame_count

    if not fall_latched:
        if current_model_label == "FALL":
            shoulder_ok, shoulder_detail = get_recent_shoulder_drop_near_ground_signal(raw_landmarks)
            hip_ok, hip_detail = get_hip_touch_ground_with_straight_legs_signal(raw_landmarks)

            if shoulder_ok or hip_ok:
                fall_latched = True
                recovery_frame_count = 0
                print(
                    f"[FALL ENTRY] shoulder_ok={shoulder_ok} ({shoulder_detail}), "
                    f"hip_ok={hip_ok} ({hip_detail})"
                )
                return "FALL"

            print(
                f"[FALL BLOCKED] model=FALL nhưng chưa đủ rule. "
                f"shoulder=({shoulder_detail}), hip=({hip_detail})"
            )
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
        pred_history.clear()
        reset_motion_context()
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