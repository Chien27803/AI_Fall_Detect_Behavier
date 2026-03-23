import numpy as np
import pandas as pd
from pathlib import Path

from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset")
NO_OF_TIMESTEPS = 30
NUM_FEATURES = 132

LABEL_MAP = {
    "ADL": 0,
    "BOXING":1,
    "FALL": 2,
}

CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]


def get_label_from_filename(file_path: Path):
    file_name = file_path.stem.upper()
    for prefix, label_value in LABEL_MAP.items():
        if file_name.startswith(prefix.upper()):
            return label_value, prefix
    return None, None


def load_csv_file(file_path: Path):
    df = pd.read_csv(file_path)

    if df.shape[1] == 133:
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

    return data


def create_samples_from_sequence(sequence, label):
    X_samples = []
    y_samples = []

    n_frames = sequence.shape[0]

    for i in range(NO_OF_TIMESTEPS, n_frames + 1):
        X_samples.append(sequence[i - NO_OF_TIMESTEPS:i, :])
        y_samples.append(label)

    return X_samples, y_samples


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

# ===== 3. Tạo sample từ train files =====
X_train, y_train = [], []
for file_path, sequence, label, prefix in train_files:
    x_part, y_part = create_samples_from_sequence(sequence, label)
    X_train.extend(x_part)
    y_train.extend(y_part)
    print(f"[TRAIN] {file_path.name} -> {prefix} ({label}), samples={len(x_part)}")

# ===== 4. Tạo sample từ test files =====
X_test, y_test = [], []
for file_path, sequence, label, prefix in test_files:
    x_part, y_part = create_samples_from_sequence(sequence, label)
    X_test.extend(x_part)
    y_test.extend(y_part)
    print(f"[TEST] {file_path.name} -> {prefix} ({label}), samples={len(x_part)}")

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

num_classes = len(CLASS_NAMES)

model = Sequential([
    Input(shape=(NO_OF_TIMESTEPS, NUM_FEATURES)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

model.save("model.h5")
print("Đã lưu model.h5")
print("Class names:", CLASS_NAMES)