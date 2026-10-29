
from pathlib import Path
import numpy as np

ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "UCI HAR Dataset"
)

ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load_split(split):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")

    folder = ROOT / split / "Inertial Signals"

    signals = [
        np.loadtxt(
            folder / f"total_acc_{axis}_{split}.txt",
            dtype=np.float32,
        )
        for axis in ("x", "y", "z")
    ]

    X = np.stack(signals, axis=-1)
    y = np.loadtxt(
        ROOT / split / f"y_{split}.txt",
        dtype=np.int64,
    ) - 1

    assert X.shape == (len(y), 128, 3), X.shape
    assert np.isfinite(X).all()
    assert np.isin(y, np.arange(6)).all()

    return X, y


def load_dataset():
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset()

    print("Dataset loaded successfully!")
    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape)
    print("X_test: ", X_test.shape, X_test.dtype)
    print("y_test: ", y_test.shape)

    print("\nTraining class counts:")
    for i, name in enumerate(ACTIVITIES):
        print(name, int(np.sum(y_train == i)))

    print("\nTest class counts:")
    for i, name in enumerate(ACTIVITIES):
        print(name, int(np.sum(y_test == i)))

    print("\nAccelerometer range:")
    print(float(X_train.min()), "to", float(X_train.max()))
