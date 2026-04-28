import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# ====== CONFIG ======
label = "ADL"   # đổi label khi thu hành động khác
subject_name = "subject2"
no_of_frames = 140      # số frame cho 1 sample (chuẩn LSTM)

countdown_seconds = 10  # sau khi bấm r, đợi 10 giây mới record

base_save_path = "dataset"
subfolder_name = "train_s2"
save_path = os.path.join(base_save_path, subfolder_name)

os.makedirs(save_path, exist_ok=True)

# ====== CAMERA ======
cap = cv2.VideoCapture(0)

# ====== MEDIAPIPE ======
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
recording = False          # đang ghi dữ liệu thật
counting_down = False      # đang đếm ngược trước khi ghi
countdown_start_time = None

prev_landmarks = None      # dùng để tính vx, vy, vz


# ====== FUNCTION ======
def extract_current_landmarks(results):
    coords = []
    for lm in results.pose_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z, lm.visibility])
    return coords


def make_landmark_timestep(results, prev_landmarks=None):
    current_landmarks = extract_current_landmarks(results)
    c_lm = []

    for i, lm in enumerate(current_landmarks):
        x, y, z, visibility = lm

        if prev_landmarks is None:
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            prev_x, prev_y, prev_z, _ = prev_landmarks[i]
            vx = x - prev_x
            vy = y - prev_y
            vz = z - prev_z

        # in ra terminal để kiểm tra
        print(x, y, z, visibility, vx, vy, vz)

        c_lm.extend([x, y, z, visibility, vx, vy, vz])

    return c_lm, current_landmarks


def draw_landmark_on_image(results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img


def get_next_filename():
    prefix = f"{label}_{subject_name}_"
    files = [f for f in os.listdir(save_path) if f.startswith(prefix) and f.endswith(".csv")]
    return f"{label}_{subject_name}_{len(files) + 1}.csv"


# ====== LOOP ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    # ====== XỬ LÝ ĐẾM NGƯỢC ======
    if counting_down:
        elapsed = time.time() - countdown_start_time
        remaining = countdown_seconds - int(elapsed)

        if elapsed >= countdown_seconds:
            print("✅ Start recording now!")

            counting_down = False
            recording = True

            lm_list = []
            prev_landmarks = None

    if results.pose_landmarks:
        frame = draw_landmark_on_image(results, frame)

        if recording:
            lm, prev_landmarks = make_landmark_timestep(results, prev_landmarks)
            lm_list.append(lm)

            # đủ frame thì lưu file
            if len(lm_list) == no_of_frames:
                df = pd.DataFrame(lm_list)
                file_name = get_next_filename()
                full_path = os.path.join(save_path, file_name)
                df.to_csv(full_path, index=False)
                print(f"✅ Saved: {full_path}")

                lm_list = []
                recording = False
                prev_landmarks = None

    else:
        # nếu mất pose trong lúc record thì reset prev_landmarks
        if recording:
            prev_landmarks = None

    # ====== HIỂN THỊ ======
    if counting_down:
        elapsed = time.time() - countdown_start_time
        remaining = max(0, countdown_seconds - int(elapsed))

        status = f"Recording starts in {remaining}s"

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            str(remaining),
            (frame.shape[1] // 2 - 40, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            5
        )

    elif recording:
        status = f"RECORDING... {len(lm_list)}/{no_of_frames}"

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        status = "Press 'r' to record"

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)

    if key == ord('r'):
        if not recording and not counting_down:
            print("⏳ Countdown started. Recording will start after 10 seconds...")
            counting_down = True
            countdown_start_time = time.time()
            lm_list = []
            prev_landmarks = None

    elif key == ord('q'):
        break

# ====== CLEANUP ======
cap.release()
cv2.destroyAllWindows()