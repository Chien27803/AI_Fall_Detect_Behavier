import cv2
import mediapipe as mp
import pandas as pd
import os

# ====== CONFIG ======
label = "ADL"   # đổi label khi thu hành động khác
subject_name = "subject6"
no_of_frames = 210      # số frame cho 1 sample (chuẩn LSTM)

base_save_path = "dataset"
subfolder_name = "train_s6"
save_path = os.path.join(base_save_path, subfolder_name)

os.makedirs(save_path, exist_ok=True)

# ====== CAMERA ======
cap = cv2.VideoCapture(0)

# ====== MEDIAPIPE ======
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
recording = False  # trạng thái ghi
prev_landmarks = None  # dùng để tính vx, vy, vz


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
    status = "RECORDING..." if recording else "Press 'r' to record"
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
        print("🎬 Start recording...")
        recording = True
        lm_list = []
        prev_landmarks = None

    elif key == ord('q'):
        break

# ====== CLEANUP ======
cap.release()
cv2.destroyAllWindows()