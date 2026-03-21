import numpy as np
import pandas as pd
from pathlib import Path

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset")
NO_OF_TIMESTEPS = 30
NUM_FEATURES = 132

LABEL_MAP = {
    "NORMAL": 0,
    "HAND WAVING": 1,
    "BODYSWING": 2,
    "FALL": 3,
}

CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

X = []
y = []

csv_files = list(DATASET_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"Không tìm thấy file csv trong: {DATASET_DIR.resolve()}")

for file_path in csv_files:
    file_name = file_path.stem.upper()

    matched_label = None
    matched_prefix = None

    for prefix, label_value in LABEL_MAP.items():
        if file_name.startswith(prefix.upper()):
            matched_label = label_value
            matched_prefix = prefix
            break

    if matched_label is None:
        print(f"[BỎ QUA] Không nhận diện được nhãn của file: {file_path.name}")
        continue

    df = pd.read_csv(file_path)

    if df.shape[1] == 133:
        dataset = df.iloc[:, 1:].values
    else:
        dataset = df.values

    if dataset.shape[1] != NUM_FEATURES:
        print(
            f"[BỎ QUA] {file_path.name} có {dataset.shape[1]} features, "
            f"cần đúng {NUM_FEATURES} features"
        )
        continue

    n_frames = dataset.shape[0]

    if n_frames < NO_OF_TIMESTEPS:
        print(
            f"[BỎ QUA] {file_path.name} có {n_frames} frame, "
            f"ít hơn NO_OF_TIMESTEPS={NO_OF_TIMESTEPS}"
        )
        continue

    count_samples = 0
    for i in range(NO_OF_TIMESTEPS, n_frames + 1):
        X.append(dataset[i - NO_OF_TIMESTEPS:i, :])
        y.append(matched_label)
        count_samples += 1

    print(
        f"[OK] {file_path.name} -> {matched_prefix} ({matched_label}), "
        f"frames={n_frames}, samples={count_samples}"
    )

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise ValueError("Không có sample hợp lệ nào trong dataset.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

num_classes = len(CLASS_NAMES)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(NO_OF_TIMESTEPS, NUM_FEATURES)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation="softmax"))

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