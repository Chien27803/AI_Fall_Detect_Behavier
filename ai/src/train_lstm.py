import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from collections import Counter

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


def get_label_from_filename(file_path: Path):
    file_name = file_path.stem.upper()
    for prefix, label_value in LABEL_MAP.items():
        if file_name.startswith(prefix.upper()):
            return label_value, prefix
    return None, None


def convert_to_relative_coordinates(sequence: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa tọa độ tuyệt đối sang tọa độ tương đối theo tâm hông.
    Mỗi landmark có 7 giá trị:
        [x, y, z, visibility, vx, vy, vz]

    Chỉ chuẩn hóa phần:
        x, y, z

    Giữ nguyên:
        visibility, vx, vy, vz

    Input:
        sequence shape = (n_frames, 231)
    Output:
        sequence shape = (n_frames, 231)
    """
    relative_sequence = sequence.copy().astype(np.float32)

    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24
    FEATURES_PER_LANDMARK = 7

    for i in range(relative_sequence.shape[0]):
        frame = relative_sequence[i].reshape(33, FEATURES_PER_LANDMARK)

        hip_center_x = (frame[LEFT_HIP_IDX, 0] + frame[RIGHT_HIP_IDX, 0]) / 2.0
        hip_center_y = (frame[LEFT_HIP_IDX, 1] + frame[RIGHT_HIP_IDX, 1]) / 2.0
        hip_center_z = (frame[LEFT_HIP_IDX, 2] + frame[RIGHT_HIP_IDX, 2]) / 2.0

        # Chỉ trừ trên x, y, z
        frame[:, 0] -= hip_center_x
        frame[:, 1] -= hip_center_y
        frame[:, 2] -= hip_center_z

        relative_sequence[i] = frame.reshape(-1)

    return relative_sequence


def load_csv_file(file_path: Path):
    df = pd.read_csv(file_path)

    # Nếu cột đầu là index thì bỏ đi
    if df.shape[1] == NUM_FEATURES + 1:
        data = df.iloc[:, 1:].values
    else:
        data = df.values

    if data.shape[1] != NUM_FEATURES:
        print(
            f"[BỎ QUA] {file_path.name} có {data.shape[1]} features, "
            f"cần đúng {NUM_FEATURES} features"
        )
        return None

    if data.shape[0] < NO_OF_TIMESTEPS:
        print(
            f"[BỎ QUA] {file_path.name} có {data.shape[0]} frame, "
            f"ít hơn NO_OF_TIMESTEPS={NO_OF_TIMESTEPS}"
        )
        return None

    data = convert_to_relative_coordinates(data)
    return data


def create_samples_from_sequence(sequence: np.ndarray, label: int):
    X_samples = []
    y_samples = []

    n_frames = sequence.shape[0]

    for i in range(NO_OF_TIMESTEPS, n_frames + 1):
        X_samples.append(sequence[i - NO_OF_TIMESTEPS:i, :])
        y_samples.append(label)

    return X_samples, y_samples


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

    file_infos.append((file_path, sequence, label, prefix))

if not file_infos:
    raise ValueError("Không có file hợp lệ nào để train.")

# ===== 2. Chia train/test theo FILE trước =====
labels_for_split = [item[2] for item in file_infos]

train_files, test_files = train_test_split(
    file_infos,
    test_size=0.2,
    random_state=42,
    stratify=labels_for_split
)

print_file_distribution("TRAIN", train_files)
print_file_distribution("TEST", test_files)

# ===== 3. Tạo sample từ train files =====
X_train, y_train = [], []
train_sample_counter = Counter()

for file_path, sequence, label, prefix in train_files:
    x_part, y_part = create_samples_from_sequence(sequence, label)
    X_train.extend(x_part)
    y_train.extend(y_part)
    train_sample_counter[label] += len(x_part)
    print(f"[TRAIN] {file_path.name} -> {prefix} ({label}), samples={len(x_part)}")

# ===== 4. Tạo sample từ test files =====
X_test, y_test = [], []
test_sample_counter = Counter()

for file_path, sequence, label, prefix in test_files:
    x_part, y_part = create_samples_from_sequence(sequence, label)
    X_test.extend(x_part)
    y_test.extend(y_part)
    test_sample_counter[label] += len(x_part)
    print(f"[TEST] {file_path.name} -> {prefix} ({label}), samples={len(x_part)}")

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
    callbacks=[checkpoint, early_stopping]
)

# Lưu model ở epoch cuối cùng
model.save("final_model.keras")

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

print("\nĐã lưu best_model.keras và final_model.keras")
print("Class names:", CLASS_NAMES)