import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from collections import Counter, defaultdict

from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

DATASET_DIR = Path("dataset")
NO_OF_TIMESTEPS = 35
NUM_FEATURES = 231   # 33 landmarks * 7 features = [x, y, z, visibility, vx, vy, vz]
LEARNING_RATE = 0.0005
EPOCHS = 15
BATCH_SIZE = 16

LABEL_MAP = {
    "ADL": 0,
    "BOXING": 1,
    "FALL": 2,
    "HAND_WAVING": 3
}

CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
NUM_LANDMARKS = 33
FEATURES_PER_LANDMARK = 7

X_IDX = 0
Y_IDX = 1
Z_IDX = 2
VIS_IDX = 3
VX_IDX = 4
VY_IDX = 5
VZ_IDX = 6

# ===== CLEANING SETTINGS =====
MIN_FRAME_VISIBILITY_MEAN = 0.35
MIN_VALID_FRAME_RATIO = 0.60
MAX_MISSING_RATIO_PER_ROW = 0.30
EPS = 1e-6

# ===== CLASS WEIGHT SETTINGS =====
CLASS_WEIGHT = {
    0: 1.0,   # ADL
    1: 1.0,   # BOXING
    2: 1.4,   # FALL
    3: 1.0    # HAND_WAVING
}


def get_label_from_filename(file_path: Path):
    file_name = file_path.stem.upper()

    for prefix, label_value in LABEL_MAP.items():
        if file_name.startswith(prefix.upper()):
            return label_value, prefix

    return None, None


def get_dataset_child_folder(file_path: Path):
    """
    Lấy tên folder con trực tiếp nằm trong dataset.

    Ví dụ:
        dataset/train_s1/ADL_001.csv       -> train_s1
        dataset/boxing_kth/BOXING_001.csv  -> boxing_kth

    Nếu file nằm sâu hơn:
        dataset/train_s1/sub/ADL_001.csv   -> train_s1
    """
    relative_path = file_path.relative_to(DATASET_DIR)

    if len(relative_path.parts) < 2:
        return "__ROOT__"

    return relative_path.parts[0]


def convert_to_relative_coordinates(sequence: np.ndarray) -> np.ndarray:
    """
    Chuyển sequence từ:
        [x, y, z, visibility, vx, vy, vz] tuyệt đối
    thành:
        [x_rel, y_rel, z_rel, visibility, vx_rel, vy_rel, vz_rel]

    Các bước:
    1) Chuẩn hóa x, y, z theo tâm hông của từng frame
    2) Tính lại vx, vy, vz từ tọa độ tương đối giữa 2 frame liên tiếp
    3) Frame đầu tiên sẽ có vx = vy = vz = 0
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2 or sequence.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Sequence phải có shape (n_frames, {NUM_FEATURES}), "
            f"nhưng nhận được {sequence.shape}"
        )

    frames = sequence.reshape(-1, NUM_LANDMARKS, FEATURES_PER_LANDMARK).copy()

    # ===== 1. Chuyển x, y, z sang tương đối theo tâm hông =====
    for i in range(frames.shape[0]):
        left_hip = frames[i, LEFT_HIP_IDX, [X_IDX, Y_IDX, Z_IDX]]
        right_hip = frames[i, RIGHT_HIP_IDX, [X_IDX, Y_IDX, Z_IDX]]
        hip_center = (left_hip + right_hip) / 2.0

        frames[i, :, X_IDX] -= hip_center[0]
        frames[i, :, Y_IDX] -= hip_center[1]
        frames[i, :, Z_IDX] -= hip_center[2]

    # ===== 2. Tính lại vx, vy, vz trên hệ tọa độ tương đối =====
    frames[:, :, VX_IDX] = 0.0
    frames[:, :, VY_IDX] = 0.0
    frames[:, :, VZ_IDX] = 0.0

    for i in range(1, frames.shape[0]):
        frames[i, :, VX_IDX] = frames[i, :, X_IDX] - frames[i - 1, :, X_IDX]
        frames[i, :, VY_IDX] = frames[i, :, Y_IDX] - frames[i - 1, :, Y_IDX]
        frames[i, :, VZ_IDX] = frames[i, :, Z_IDX] - frames[i - 1, :, Z_IDX]

    return frames.reshape(-1, NUM_FEATURES)


def clean_dataframe(df: pd.DataFrame, file_path: Path):
    """
    Làm sạch DataFrame:
    - bỏ cột index rác
    - ép kiểu số
    - thay inf -> NaN
    - loại dòng rỗng / dòng lỗi nặng
    - nội suy NaN
    """
    original_rows = len(df)

    # ===== 1. Bỏ cột index rác nếu có =====
    if df.shape[1] == NUM_FEATURES + 1:
        first_col_name = str(df.columns[0]).strip().lower()

        if first_col_name.startswith("unnamed") or first_col_name in ["index", "frame"]:
            df = df.iloc[:, 1:]
        else:
            # nếu dư đúng 1 cột thì vẫn bỏ cột đầu
            df = df.iloc[:, 1:]

    # ===== 2. Kiểm tra số cột =====
    if df.shape[1] != NUM_FEATURES:
        print(
            f"[BỎ QUA] {file_path.name} có {df.shape[1]} features, "
            f"cần đúng {NUM_FEATURES} features"
        )
        return None, None

    # ===== 3. Ép toàn bộ sang số =====
    df = df.apply(pd.to_numeric, errors="coerce")

    # ===== 4. Thay inf/-inf thành NaN =====
    df = df.replace([np.inf, -np.inf], np.nan)

    # ===== 5. Bỏ dòng rỗng hoàn toàn =====
    df = df.dropna(how="all")

    if len(df) == 0:
        print(f"[BỎ QUA] {file_path.name} không còn dữ liệu sau khi bỏ dòng rỗng")
        return None, None

    # ===== 6. Bỏ dòng bị thiếu quá nhiều =====
    row_missing_ratio = df.isna().mean(axis=1)
    df = df.loc[row_missing_ratio <= MAX_MISSING_RATIO_PER_ROW].copy()

    if len(df) == 0:
        print(f"[BỎ QUA] {file_path.name} toàn bộ frame bị lỗi nặng")
        return None, None

    # ===== 7. Nội suy dữ liệu thiếu =====
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.ffill().bfill()

    # ===== 8. Kiểm tra lại =====
    if df.isnull().values.any():
        print(f"[BỎ QUA] {file_path.name} vẫn còn NaN sau khi làm sạch")
        return None, None

    return df, original_rows


def filter_low_visibility_frames(data: np.ndarray, file_path: Path, original_rows: int):
    """
    Lọc frame có chất lượng thấp dựa trên visibility trung bình của 33 landmarks.
    """
    if data.ndim != 2 or data.shape[1] != NUM_FEATURES:
        print(f"[BỎ QUA] {file_path.name} shape không hợp lệ trước bước lọc visibility: {data.shape}")
        return None

    frames = data.reshape(-1, NUM_LANDMARKS, FEATURES_PER_LANDMARK)

    visibility_values = frames[:, :, VIS_IDX]
    frame_visibility_mean = np.mean(visibility_values, axis=1)

    valid_mask = frame_visibility_mean >= MIN_FRAME_VISIBILITY_MEAN
    filtered_frames = frames[valid_mask]

    kept_count = filtered_frames.shape[0]
    kept_ratio = kept_count / max(original_rows, 1)

    if kept_count < NO_OF_TIMESTEPS:
        print(
            f"[BỎ QUA] {file_path.name} sau lọc visibility chỉ còn {kept_count} frame, "
            f"ít hơn NO_OF_TIMESTEPS={NO_OF_TIMESTEPS}"
        )
        return None

    if kept_ratio < MIN_VALID_FRAME_RATIO:
        print(
            f"[BỎ QUA] {file_path.name} chỉ giữ được {kept_count}/{original_rows} frame "
            f"({kept_ratio:.2%}), dưới ngưỡng {MIN_VALID_FRAME_RATIO:.0%}"
        )
        return None

    return filtered_frames.reshape(-1, NUM_FEATURES).astype(np.float32)


def load_csv_file(file_path: Path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[BỎ QUA] Không đọc được file {file_path.name}: {e}")
        return None

    # ===== 1. Làm sạch DataFrame =====
    cleaned_df, original_rows = clean_dataframe(df, file_path)

    if cleaned_df is None:
        return None

    # ===== 2. Chuyển sang numpy =====
    try:
        data = cleaned_df.to_numpy(dtype=np.float32)
    except Exception as e:
        print(f"[BỎ QUA] {file_path.name} lỗi khi chuyển sang numpy float32: {e}")
        return None

    # ===== 3. Kiểm tra số frame trước khi lọc visibility =====
    if data.shape[0] < NO_OF_TIMESTEPS:
        print(
            f"[BỎ QUA] {file_path.name} có {data.shape[0]} frame, "
            f"ít hơn NO_OF_TIMESTEPS={NO_OF_TIMESTEPS}"
        )
        return None

    # ===== 4. Lọc frame chất lượng thấp theo visibility =====
    data = filter_low_visibility_frames(data, file_path, original_rows)

    if data is None:
        return None

    # ===== 5. Chuẩn hóa sang tọa độ tương đối =====
    try:
        data = convert_to_relative_coordinates(data)
    except Exception as e:
        print(f"[BỎ QUA] {file_path.name} lỗi khi chuẩn hóa dữ liệu: {e}")
        return None

    return data


def create_samples_from_sequence(sequence: np.ndarray, label: int):
    X_samples = []
    y_samples = []

    n_frames = sequence.shape[0]

    for i in range(NO_OF_TIMESTEPS, n_frames + 1):
        X_samples.append(sequence[i - NO_OF_TIMESTEPS:i, :])
        y_samples.append(label)

    return X_samples, y_samples


def split_files_by_child_folder_and_label(file_infos, test_size=0.2, random_state=42):
    """
    Chia train/test theo từng folder con và từng label trong folder đó.

    Nghĩa là:
        dataset/train_s1/ADL_*.csv  -> 80% TRAIN, 20% TEST
        dataset/train_s1/FALL_*.csv -> 80% TRAIN, 20% TEST
        dataset/train_s2/ADL_*.csv  -> 80% TRAIN, 20% TEST
        ...

    file_infos item gồm:
        (file_path, sequence, label, prefix, child_folder)
    """
    files_by_folder_and_label = defaultdict(list)

    for item in file_infos:
        file_path, sequence, label, prefix, child_folder = item
        key = (child_folder, label)
        files_by_folder_and_label[key].append(item)

    train_files = []
    test_files = []

    print("\n===== CHIA TRAIN/TEST THEO FOLDER CON + LABEL =====")

    for (folder_name, label), group_files in sorted(files_by_folder_and_label.items()):
        class_name = CLASS_NAMES[label]
        total_files = len(group_files)

        if total_files == 1:
            train_files.extend(group_files)
            print(
                f"{folder_name} | {class_name}: total=1 -> TRAIN=1, TEST=0 "
                f"(vì chỉ có 1 file)"
            )
            continue

        group_train, group_test = train_test_split(
            group_files,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        train_files.extend(group_train)
        test_files.extend(group_test)

        print(
            f"{folder_name} | {class_name}: total={total_files}, "
            f"TRAIN={len(group_train)}, TEST={len(group_test)}"
        )

    return train_files, test_files


def print_file_distribution(title, files):
    print(f"\n===== {title} - SỐ FILE THEO CLASS =====")
    counter = Counter()

    for _, _, label, _, _ in files:
        counter[label] += 1

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {counter[class_id]} files")


def print_folder_distribution(title, files):
    print(f"\n===== {title} - SỐ FILE THEO FOLDER =====")
    counter = Counter()

    for _, _, _, _, child_folder in files:
        counter[child_folder] += 1

    for folder_name in sorted(counter.keys()):
        print(f"{folder_name}: {counter[folder_name]} files")


def print_folder_label_distribution(title, files):
    print(f"\n===== {title} - SỐ FILE THEO FOLDER + LABEL =====")
    counter = Counter()

    for _, _, label, _, child_folder in files:
        counter[(child_folder, label)] += 1

    for child_folder, label in sorted(counter.keys()):
        class_name = CLASS_NAMES[label]
        print(f"{child_folder} | {class_name}: {counter[(child_folder, label)]} files")


def print_sample_distribution(title, sample_counter):
    print(f"\n===== {title} - SỐ SAMPLE THEO CLASS =====")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {sample_counter[class_id]} samples")


# ===== 1. Lấy danh sách file hợp lệ =====
file_infos = []

csv_files = list(DATASET_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"Không tìm thấy file csv trong: {DATASET_DIR.resolve()}")

for file_path in csv_files:
    label, prefix = get_label_from_filename(file_path)

    if label is None:
        print(f"[BỎ QUA] Không nhận diện được nhãn của file: {file_path.name}")
        continue

    sequence = load_csv_file(file_path)

    if sequence is None:
        continue

    child_folder = get_dataset_child_folder(file_path)

    file_infos.append((file_path, sequence, label, prefix, child_folder))

if not file_infos:
    raise ValueError("Không có file hợp lệ nào để train.")


# ===== 2. Chia train/test theo từng FOLDER CON + từng LABEL =====
train_files, test_files = split_files_by_child_folder_and_label(
    file_infos,
    test_size=0.2,
    random_state=42
)

if not train_files:
    raise ValueError("Không có file TRAIN nào sau khi chia dữ liệu.")

if not test_files:
    raise ValueError(
        "Không có file TEST nào sau khi chia dữ liệu. "
        "Có thể mỗi nhóm folder + label chỉ có 1 file hợp lệ."
    )

print_file_distribution("TRAIN", train_files)
print_file_distribution("TEST", test_files)

print_folder_distribution("TRAIN", train_files)
print_folder_distribution("TEST", test_files)

print_folder_label_distribution("TRAIN", train_files)
print_folder_label_distribution("TEST", test_files)


# ===== 3. Tạo sample từ train files =====
X_train, y_train = [], []
train_sample_counter = Counter()

for file_path, sequence, label, prefix, child_folder in train_files:
    x_part, y_part = create_samples_from_sequence(sequence, label)

    X_train.extend(x_part)
    y_train.extend(y_part)
    train_sample_counter[label] += len(x_part)

    print(
        f"[TRAIN] folder={child_folder} | {file_path.name} "
        f"-> {prefix} ({label}), samples={len(x_part)}"
    )


# ===== 4. Tạo sample từ test files =====
X_test, y_test = [], []
test_sample_counter = Counter()

for file_path, sequence, label, prefix, child_folder in test_files:
    x_part, y_part = create_samples_from_sequence(sequence, label)

    X_test.extend(x_part)
    y_test.extend(y_part)
    test_sample_counter[label] += len(x_part)

    print(
        f"[TEST] folder={child_folder} | {file_path.name} "
        f"-> {prefix} ({label}), samples={len(x_part)}"
    )

print_sample_distribution("TRAIN", train_sample_counter)
print_sample_distribution("TEST", test_sample_counter)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

print("\n===== SHAPE DỮ LIỆU =====")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print(f"Learning rate: {LEARNING_RATE}")

print("\n===== CLASS WEIGHT =====")

for class_id, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name}: {CLASS_WEIGHT[class_id]}")

num_classes = len(CLASS_NAMES)


# ===== 5. Build model =====
model = Sequential([
    Input(shape=(NO_OF_TIMESTEPS, NUM_FEATURES)),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ===== 6. Callbacks =====
checkpoint = ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
    verbose=1
)


# ===== 7. Train =====
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping],
    class_weight=CLASS_WEIGHT
)


# ===== 8. Load best model để đánh giá =====
best_model = tf.keras.models.load_model("best_model.keras")

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


# ===== 9. Confusion Matrix + Classification Report =====
y_pred_probs = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

print("\n===== CONFUSION MATRIX =====")

cm_df = pd.DataFrame(
    cm,
    index=[f"TRUE_{c}" for c in CLASS_NAMES],
    columns=[f"PRED_{c}" for c in CLASS_NAMES]
)

print(cm_df)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

print("\nĐã lưu best_model.keras")
print("Class names:", CLASS_NAMES) 