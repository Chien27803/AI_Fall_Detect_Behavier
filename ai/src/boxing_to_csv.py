import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

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

        c_lm.extend([x, y, z, visibility, vx, vy, vz])

    return c_lm, current_landmarks


def get_subject_from_video_name(video_path: Path):
    """
    Ví dụ:
        person01_boxing_d1_uncomp.avi -> subject1
        person02_boxing_d3_uncomp.avi -> subject2
    """
    file_name = video_path.stem.lower()
    match = re.search(r"person(\d+)", file_name)
    if not match:
        return None

    person_number = int(match.group(1))
    return f"subject{person_number}"


def build_output_filename(subject_name: str, index: int):
    return f"{LABEL}_{subject_name}_{index:02d}.csv"


def extract_video_to_csv(video_path: Path, output_filename: str):
    cap = cv2.VideoCapture(str(video_path))
    lm_list = []
    total_frames = 0
    valid_frames = 0
    prev_landmarks = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm, prev_landmarks = make_landmark_timestep(results, prev_landmarks)
            lm_list.append(lm)
            valid_frames += 1

    cap.release()

    if not lm_list:
        print(f"[BỎ QUA] {video_path.name}: không lấy được landmark nào")
        return

    df = pd.DataFrame(lm_list)
    output_file = OUTPUT_DIR / output_filename
    df.to_csv(output_file, index=False)

    print(
        f"[OK] {video_path.name} -> {output_file.name} | "
        f"label={LABEL}, total_frames={total_frames}, valid_frames={valid_frames}"
    )


def main():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy folder video: {VIDEO_DIR.resolve()}")

    video_files = sorted(VIDEO_DIR.rglob("*.avi"), key=lambda x: x.name)
    if not video_files:
        raise FileNotFoundError("Không tìm thấy file .avi nào trong folder boxing.")

    print(f"Tìm thấy {len(video_files)} video boxing")

    # Gom video theo subject
    subject_to_videos = defaultdict(list)

    for video_path in video_files:
        subject_name = get_subject_from_video_name(video_path)
        if subject_name is None:
            print(f"[BỎ QUA] Không xác định được subject từ file: {video_path.name}")
            continue
        subject_to_videos[subject_name].append(video_path)

    if not subject_to_videos:
        raise ValueError("Không có video boxing hợp lệ để xử lý.")

    # Với mỗi subject, đánh số 01, 02, 03...
    for subject_name in sorted(subject_to_videos.keys()):
        subject_videos = sorted(subject_to_videos[subject_name], key=lambda x: x.name)

        for idx, video_path in enumerate(subject_videos, start=1):
            output_filename = build_output_filename(subject_name, idx)
            extract_video_to_csv(video_path, output_filename)

    print("Hoàn tất chuyển avi -> csv cho BOXING")


if __name__ == "__main__":
    main()