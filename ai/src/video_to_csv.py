import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# ====== PATH ======
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

VIDEO_DIR = BASE_DIR / "data" / "raw"

# chọn subject muốn convert
SUBJECTS_TO_PROCESS = ["subject4"]   # đổi thành ["subject1"] khi làm train
OUTPUT_SUBFOLDER = "train_s4"            # đổi thành "train" khi làm train

OUTPUT_DIR = SRC_DIR / "dataset" / OUTPUT_SUBFOLDER
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_LABELS = {"ADL", "FALL"}

# ====== MEDIAPIPE ======
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm


def extract_label_from_path(video_path: Path):
    parent_folder = video_path.parent.name.upper()
    if parent_folder in VALID_LABELS:
        return parent_folder
    return None


def build_output_filename(video_path: Path, label: str):
    subject_name = video_path.parent.parent.name
    video_name = video_path.stem
    return f"{label}_{subject_name}_{video_name}.csv"


def extract_video_to_csv(video_path: Path):
    label = extract_label_from_path(video_path)

    if label is None:
        print(f"[BỎ QUA] {video_path}: không xác định được nhãn")
        return

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
    output_file = OUTPUT_DIR / build_output_filename(video_path, label)
    df.to_csv(output_file, index=False)

    print(
        f"[OK] {video_path.name} -> {output_file.name} | "
        f"label={label}, total_frames={total_frames}, valid_frames={valid_frames}"
    )


def main():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy folder video: {VIDEO_DIR.resolve()}")

    video_files = []
    for subject in SUBJECTS_TO_PROCESS:
        subject_dir = VIDEO_DIR / subject
        if subject_dir.exists():
            video_files.extend(subject_dir.rglob("*.mp4"))
        else:
            print(f"[CẢNH BÁO] Không tìm thấy: {subject_dir}")

    if not video_files:
        raise FileNotFoundError("Không tìm thấy file mp4 nào cho subject đã chọn.")

    print(f"Tìm thấy {len(video_files)} video từ {SUBJECTS_TO_PROCESS}")

    for video_path in video_files:
        extract_video_to_csv(video_path)

    print("Hoàn tất chuyển mp4 -> csv")


if __name__ == "__main__":
    main()