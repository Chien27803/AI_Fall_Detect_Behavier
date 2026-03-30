import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# ====== PATH ======
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

# folder chứa các folder ảnh fall
IMAGE_ROOT_DIR = BASE_DIR / "data" / "raw" / "subject5"
OUTPUT_SUBFOLDER = "train_s5"

OUTPUT_DIR = SRC_DIR / "dataset" / OUTPUT_SUBFOLDER
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL = "FALL"
SUBJECT_NAME = "subject5"

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


def get_sorted_image_files(folder_path: Path):
    image_files = [p for p in folder_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(image_files, key=lambda x: x.name)


def build_output_filename(index: int):
    return f"{LABEL}_{SUBJECT_NAME}_{index:02d}.csv"


def extract_image_folder_to_csv(image_folder: Path, index: int):
    image_files = get_sorted_image_files(image_folder)

    if not image_files:
        print(f"[BỎ QUA] {image_folder.name}: không có ảnh hợp lệ")
        return

    lm_list = []
    total_frames = 0
    valid_frames = 0
    prev_landmarks = None

    for image_path in image_files:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm, prev_landmarks = make_landmark_timestep(results, prev_landmarks)
            lm_list.append(lm)
            valid_frames += 1

    if not lm_list:
        print(f"[BỎ QUA] {image_folder.name}: không lấy được landmark nào")
        return

    df = pd.DataFrame(lm_list)
    output_file = OUTPUT_DIR / build_output_filename(index)
    df.to_csv(output_file, index=False)

    print(
        f"[OK] {image_folder.name} -> {output_file.name} | "
        f"label={LABEL}, total_frames={total_frames}, valid_frames={valid_frames}"
    )


def main():
    if not IMAGE_ROOT_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {IMAGE_ROOT_DIR.resolve()}")

    # lấy tất cả folder kiểu fall-01-cam0-rgb, fall-02-cam0-rgb...
    image_folders = [p for p in IMAGE_ROOT_DIR.iterdir() if p.is_dir() and "fall" in p.name.lower()]

    if not image_folders:
        raise FileNotFoundError("Không tìm thấy folder ảnh FALL nào.")

    image_folders = sorted(image_folders, key=lambda x: x.name)

    print(f"Tìm thấy {len(image_folders)} folder ảnh FALL")

    for idx, image_folder in enumerate(image_folders, start=1):
        extract_image_folder_to_csv(image_folder, idx)

    print("Hoàn tất chuyển folder ảnh -> csv")


if __name__ == "__main__":
    main()