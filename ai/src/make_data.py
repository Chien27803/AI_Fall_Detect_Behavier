import cv2
import mediapipe as mp
import pandas as pd
import os

# ====== CONFIG ======
label = "BOXING"   # đổi label khi thu hành động khác
no_of_frames = 150     # số frame cho 1 sample (chuẩn LSTM)
save_path = "dataset" # thư mục lưu data
os.makedirs(save_path, exist_ok=True)

# ====== CAMERA ======
cap = cv2.VideoCapture(0)

# ====== MEDIAPIPE ======
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
recording = False  # trạng thái ghi

# ====== FUNCTION ======
def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        print(lm.x, lm.y, lm.z, lm.visibility)  # 🔥 in ra terminal
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm


def draw_landmark_on_image(results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img


def get_next_filename():
    files = [f for f in os.listdir(save_path) if f.startswith(label)]
    return f"{label}_{len(files)+1}.csv"


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
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            # đủ frame thì lưu file
            if len(lm_list) == no_of_frames:
                df = pd.DataFrame(lm_list)
                file_name = get_next_filename()
                df.to_csv(os.path.join(save_path, file_name), index=False)
                print(f"✅ Saved: {file_name}")

                lm_list = []
                recording = False

    # ====== HIỂN THỊ ======
    status = "RECORDING..." if recording else "Press 'r' to record"
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)

    if key == ord('r'):
        print("🎬 Start recording...")
        recording = True
        lm_list = []

    elif key == ord('q'):
        break

# ====== CLEANUP ======
cap.release()
cv2.destroyAllWindows()