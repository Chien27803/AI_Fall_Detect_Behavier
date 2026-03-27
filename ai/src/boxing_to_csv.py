import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# ====== PATH ======
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

VIDEO_DIR = BASE_DIR / "data" / "raw" / "boxing"
OUTPUT_SUBFOLDER = "boxing_kth"

OUTPUT_DIR = SRC_DIR / "dataset" / OUTPUT_SUBFOLDER
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL = "BOXING"

# ====== MEDIAPIPE ======
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm


def build_output_filename(video_path: Path):
    return f"{LABEL}_{video_path.stem}.csv"


def extract_video_to_csv(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    lm_list = []
    total_frames = 0
    valid_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            valid_frames += 1

    cap.release()

    if not lm_list:
        print(f"[BỎ QUA] {video_path.name}: không lấy được landmark nào")
        return

    df = pd.DataFrame(lm_list)
    output_file = OUTPUT_DIR / build_output_filename(video_path)
    df.to_csv(output_file, index=False)

    print(
        f"[OK] {video_path.name} -> {output_file.name} | "
        f"label={LABEL}, total_frames={total_frames}, valid_frames={valid_frames}"
    )


def main():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy folder video: {VIDEO_DIR.resolve()}")

    video_files = list(VIDEO_DIR.rglob("*.avi"))
    if not video_files:
        raise FileNotFoundError("Không tìm thấy file .avi nào trong folder boxing.")

    print(f"Tìm thấy {len(video_files)} video boxing")

    for video_path in video_files:
        extract_video_to_csv(video_path)

    print("Hoàn tất chuyển avi -> csv cho BOXING")


if __name__ == "__main__":
    main()