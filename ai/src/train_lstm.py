import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =========================================================
# CONFIG
# =========================================================
DATASET_DIR = Path("dataset")

NO_OF_TIMESTEPS = 35
NUM_FEATURES = 231                 # 33 landmarks * 7 = [x, y, z, visibility, vx, vy, vz]
FEATURES_PER_LANDMARK = 7

WINDOW_STRIDE = 1                  # giữ nhiều sample hơn để model học tốt hơn
TEST_SIZE = 0.20                   # 20% file cho test
VAL_SIZE_FROM_TRAINVAL = 0.125     # lấy 12.5% của train_val => ~10% tổng
RANDOM_STATE = 42

LEARNING_RATE = 5e-4
EPOCHS = 25                        # mức trần, early stopping sẽ dừng sớm nếu cần
BATCH_SIZE = 16

MAX_NAN_RATIO = 0.20
MIN_STD = 1e-6
DUP_FRAME_EPS = 1e-8
SMOOTHING_WINDOW = 1               # 1 = tắt smoothing

LABEL_MAP = {
    "ADL": 0,
    "BOXING": 1,
    "FALL": 2,
    "HAND_WAVING": 3
}

CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================================================
# UTILS
# =========================================================
def build_feature_indices(local_feature_ids):
    cols = []
    for lm in range(33):
        base = lm * FEATURES_PER_LANDMARK
        for fid in local_feature_ids:
            cols.append(base + fid)
    return cols


XYZ_COLS = build_feature_indices([0, 1, 2])
VIS_COLS = build_feature_indices([3])
VEL_COLS = build_feature_indices([4, 5, 6])


def get_label_from_filename(file_path: Path):
    """
    Lấy label từ prefix tên file.
    Ví dụ:
        ADL_xxx.csv -> ADL
        FALL_01.csv -> FALL
    """
    file_name = file_path.stem.upper()
    for prefix, label_value in LABEL_MAP.items():
        if file_name.startswith(prefix.upper()):
            return label_value, prefix
    return None, None


def can_stratify(file_list):
    counts = Counter(item[2] for item in file_list)
    return all(count >= 2 for count in counts.values())


def safe_split(file_list, test_size, random_state, split_name):
    """
    Split theo file.
    Nếu mỗi class không đủ file để stratify thì fallback sang random split.
    """
    labels = [item[2] for item in file_list]
    stratify = labels if can_stratify(file_list) else None

    if stratify is None:
        print(f"[CẢNH BÁO] {split_name}: không đủ file mỗi class để stratify, chuyển sang random split.")

    return train_test_split(
        file_list,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )


def print_file_distribution(title, files):
    print(f"\n===== {title} - SỐ FILE THEO CLASS =====")
    counter = Counter()
    for _, _, label, _ in files:
        counter[label] += 1

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {counter[class_id]} files")


def print_sample_distribution(title, sample_counter):
    print(f"\n===== {title} - SỐ SAMPLE THEO CLASS =====")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {sample_counter[class_id]} samples")


# =========================================================
# CLEANING + PREPROCESS
# =========================================================
def remove_consecutive_duplicate_frames(sequence: np.ndarray, eps: float = DUP_FRAME_EPS) -> np.ndarray:
    """
    Bỏ frame trùng liên tiếp gần như hoàn toàn.
    """
    if sequence.shape[0] <= 1:
        return sequence

    keep_indices = [0]
    for i in range(1, sequence.shape[0]):
        max_abs_diff = np.max(np.abs(sequence[i] - sequence[i - 1]))
        if max_abs_diff > eps:
            keep_indices.append(i)

    return sequence[keep_indices]


def smooth_sequence(sequence: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """
    Smoothing nhẹ nếu cần.
    window=1 thì giữ nguyên.
    """
    if window <= 1 or sequence.shape[0] < window:
        return sequence.astype(np.float32)

    df = pd.DataFrame(sequence)
    smoothed = (
        df.rolling(window=window, min_periods=1, center=True)
          .mean()
          .to_numpy(dtype=np.float32)
    )

    smoothed[:, VIS_COLS] = np.clip(smoothed[:, VIS_COLS], 0.0, 1.0)
    return smoothed


def convert_to_relative_coordinates(sequence: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa x, y, z theo tâm hông.
    Giữ nguyên visibility, vx, vy, vz.
    """
    relative_sequence = sequence.copy().astype(np.float32)

    for i in range(relative_sequence.shape[0]):
        frame = relative_sequence[i].reshape(33, FEATURES_PER_LANDMARK)

        hip_center_x = (frame[LEFT_HIP_IDX, 0] + frame[RIGHT_HIP_IDX, 0]) / 2.0
        hip_center_y = (frame[LEFT_HIP_IDX, 1] + frame[RIGHT_HIP_IDX, 1]) / 2.0
        hip_center_z = (frame[LEFT_HIP_IDX, 2] + frame[RIGHT_HIP_IDX, 2]) / 2.0

        frame[:, 0] -= hip_center_x
        frame[:, 1] -= hip_center_y
        frame[:, 2] -= hip_center_z

        relative_sequence[i] = frame.reshape(-1)

    return relative_sequence


def load_csv_file(file_path: Path):
    """
    Đọc CSV và làm sạch dữ liệu.
    """
    df = pd.read_csv(file_path)

    # Nếu cột đầu là index thì bỏ đi
    if df.shape[1] == NUM_FEATURES + 1:
        df = df.iloc[:, 1:]

    if df.shape[1] != NUM_FEATURES:
        print(f"[BỎ QUA] {file_path.name} có {df.shape[1]} features, cần đúng {NUM_FEATURES}")
        return None

    # Ép numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    nan_ratio = df.isna().sum().sum() / max(df.size, 1)
    if nan_ratio > MAX_NAN_RATIO:
        print(f"[BỎ QUA] {file_path.name} có tỷ lệ NaN quá cao: {nan_ratio:.2%}")
        return None

    # Nội suy và fill
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.ffill().bfill().fillna(0.0)

    data = df.to_numpy(dtype=np.float32)

    # Visibility về [0, 1]
    data[:, VIS_COLS] = np.clip(data[:, VIS_COLS], 0.0, 1.0)

    # Bỏ frame trùng liên tiếp
    before = data.shape[0]
    data = remove_consecutive_duplicate_frames(data)
    removed = before - data.shape[0]
    if removed > 0:
        print(f"[CLEAN] {file_path.name}: đã bỏ {removed} frame trùng liên tiếp")

    if data.shape[0] < NO_OF_TIMESTEPS:
        print(f"[BỎ QUA] {file_path.name} có {data.shape[0]} frame, ít hơn NO_OF_TIMESTEPS={NO_OF_TIMESTEPS}")
        return None

    # Relative coordinates
    data = convert_to_relative_coordinates(data)

    # Smoothing nếu cần
    data = smooth_sequence(data, window=SMOOTHING_WINDOW)

    if not np.isfinite(data).all():
        print(f"[BỎ QUA] {file_path.name} còn giá trị không hợp lệ sau cleaning")
        return None

    return data


# =========================================================
# SAMPLE CREATION
# =========================================================
def create_samples_from_sequence(sequence: np.ndarray, label: int, stride: int = WINDOW_STRIDE):
    X_samples = []
    y_samples = []

    n_frames = sequence.shape[0]

    end_positions = list(range(NO_OF_TIMESTEPS, n_frames + 1, stride))

    # đảm bảo có window cuối cùng
    if len(end_positions) == 0 or end_positions[-1] != n_frames:
        end_positions.append(n_frames)

    for end_idx in end_positions:
        start_idx = end_idx - NO_OF_TIMESTEPS
        X_samples.append(sequence[start_idx:end_idx, :])
        y_samples.append(label)

    return X_samples, y_samples


def build_dataset_from_files(files, split_name):
    X, y = [], []
    sample_counter = Counter()

    for file_path, sequence, label, prefix in files:
        x_part, y_part = create_samples_from_sequence(sequence, label, stride=WINDOW_STRIDE)
        X.extend(x_part)
        y.extend(y_part)
        sample_counter[label] += len(x_part)
        print(f"[{split_name}] {file_path.name} -> {prefix} ({label}), samples={len(x_part)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, sample_counter


# =========================================================
# SCALE THEO TRAIN ONLY
# =========================================================
def fit_feature_scaler(X_train: np.ndarray):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < MIN_STD] = 1.0
    return mean, std


def apply_feature_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)


# =========================================================
# LOAD VALID FILES
# =========================================================
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

    file_infos.append((file_path, sequence, label, prefix))

if not file_infos:
    raise ValueError("Không có file hợp lệ nào để train.")

print(f"\nTổng số file hợp lệ: {len(file_infos)}")

# =========================================================
# SPLIT THEO FILE
# train_val / test
# rồi train / val
# =========================================================
train_val_files, test_files = safe_split(
    file_infos,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    split_name="train_test_split"
)

train_files, val_files = safe_split(
    train_val_files,
    test_size=VAL_SIZE_FROM_TRAINVAL,
    random_state=RANDOM_STATE,
    split_name="train_val_split"
)

if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
    raise ValueError("Một trong các tập train/val/test bị rỗng. Hãy tăng số file hoặc chỉnh lại tỷ lệ split.")

print_file_distribution("TRAIN", train_files)
print_file_distribution("VAL", val_files)
print_file_distribution("TEST", test_files)

# =========================================================
# BUILD SAMPLES
# =========================================================
X_train, y_train, train_sample_counter = build_dataset_from_files(train_files, "TRAIN")
X_val, y_val, val_sample_counter = build_dataset_from_files(val_files, "VAL")
X_test, y_test, test_sample_counter = build_dataset_from_files(test_files, "TEST")

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError("Một trong các tập sample train/val/test bị rỗng. Hãy kiểm tra NO_OF_TIMESTEPS hoặc dữ liệu đầu vào.")

print_sample_distribution("TRAIN", train_sample_counter)
print_sample_distribution("VAL", val_sample_counter)
print_sample_distribution("TEST", test_sample_counter)

print("\n===== SHAPE DỮ LIỆU TRƯỚC SCALE =====")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# =========================================================
# SCALE THEO TRAIN ONLY
# =========================================================
feature_mean, feature_std = fit_feature_scaler(X_train)

X_train = apply_feature_scaler(X_train, feature_mean, feature_std)
X_val = apply_feature_scaler(X_val, feature_mean, feature_std)
X_test = apply_feature_scaler(X_test, feature_mean, feature_std)

# Lưu scaler để dùng cho inference nếu cần
np.savez(
    "feature_scaler.npz",
    mean=feature_mean,
    std=feature_std
)

print("\n===== PARAMS =====")
print(f"NO_OF_TIMESTEPS: {NO_OF_TIMESTEPS}")
print(f"WINDOW_STRIDE: {WINDOW_STRIDE}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"EPOCHS: {EPOCHS}")
print(f"SMOOTHING_WINDOW: {SMOOTHING_WINDOW}")

num_classes = len(CLASS_NAMES)

# =========================================================
# MODEL
# =========================================================
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

print("\n===== MODEL SUMMARY =====")
model.summary()

# =========================================================
# CALLBACKS
# =========================================================
checkpoint = ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# =========================================================
# TRAIN
# =========================================================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    shuffle=True,
    verbose=1
)

# Lưu model cuối cùng
model.save("final_model.keras")

# =========================================================
# EVALUATE ON TEST ONLY
# =========================================================
best_model = tf.keras.models.load_model("best_model.keras")

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# =========================================================
# CONFUSION MATRIX + CLASSIFICATION REPORT
# =========================================================
y_pred_probs = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

labels = list(range(num_classes))

cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\n===== CONFUSION MATRIX =====")
cm_df = pd.DataFrame(
    cm,
    index=[f"TRUE_{c}" for c in CLASS_NAMES],
    columns=[f"PRED_{c}" for c in CLASS_NAMES]
)
print(cm_df)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
))

print("\nĐã lưu:")
print("- best_model.keras")
print("- final_model.keras")
print("- feature_scaler.npz")
print("Class names:", CLASS_NAMES)