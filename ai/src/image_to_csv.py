import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# ====== PATH ======
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

IMAGE_ROOT_DIR = BASE_DIR / "data" / "raw" / "subject8"
OUTPUT_SUBFOLDER = "train_s8"

OUTPUT_DIR = SRC_DIR / "dataset" / OUTPUT_SUBFOLDER
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUBJECT_NAME = "subject8"

# ====== MEDIAPIPE ======
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ====== LABEL AUTO ======
def get_label_from_folder(folder_name: str):
    name = folder_name.lower()

    if "fall" in name:
        return "FALL"
    elif "adl" in name:
        return "ADL"
    else:
        return "UNKNOWN"

# ====== LANDMARK ======
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


def build_output_filename(label: str, index: int):
    return f"{label}_{SUBJECT_NAME}_{index:02d}.csv"


def extract_image_folder_to_csv(image_folder: Path, index: int):
    label = get_label_from_folder(image_folder.name)

    if label == "UNKNOWN":
        print(f"[BỎ QUA] {image_folder.name}: không xác định label")
        return

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
    output_file = OUTPUT_DIR / build_output_filename(label, index)
    df.to_csv(output_file, index=False)

    print(
        f"[OK] {image_folder.name} -> {output_file.name} | "
        f"label={label}, total_frames={total_frames}, valid_frames={valid_frames}"
    )


def main():
    if not IMAGE_ROOT_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {IMAGE_ROOT_DIR.resolve()}")

    # lấy cả FALL + ADL
    image_folders = [
        p for p in IMAGE_ROOT_DIR.iterdir()
        if p.is_dir() and any(k in p.name.lower() for k in ["fall", "adl"])
    ]

    if not image_folders:
        raise FileNotFoundError("Không tìm thấy folder FALL hoặc ADL.")

    image_folders = sorted(image_folders, key=lambda x: x.name)

    print(f"Tìm thấy {len(image_folders)} folder (FALL + ADL)")

    for idx, image_folder in enumerate(image_folders, start=1):
        extract_image_folder_to_csv(image_folder, idx)

    print("Hoàn tất chuyển folder ảnh -> csv")


if __name__ == "__main__":
    main()