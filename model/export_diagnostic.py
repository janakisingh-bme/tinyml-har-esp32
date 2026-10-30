
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# -------------------------------------------------
# 1. PROJECT PATHS
# Run this script from the repository root.
# -------------------------------------------------

ROOT = Path.cwd()

DATA_ROOT = ROOT / "data" / "UCI HAR Dataset"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results" / "diagnostics"

MODEL_PATH = MODEL_DIR / "har_cnn_int8.tflite"
NORM_PATH = MODEL_DIR / "normalization.npz"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 2. LOAD OFFICIAL UCI HAR TEST DATA
# -------------------------------------------------

def load_split(split):
    signal_dir = DATA_ROOT / split / "Inertial Signals"

    signals = [
        np.loadtxt(
            signal_dir / f"total_acc_{axis}_{split}.txt",
            dtype=np.float32
        )
        for axis in ["x", "y", "z"]
    ]

    X = np.stack(signals, axis=-1)

    y = np.loadtxt(
        DATA_ROOT / split / f"y_{split}.txt",
        dtype=np.int64
    ) - 1

    return X, y


X_test, y_test = load_split("test")

print("Original test shape:", X_test.shape)
print("Labels shape:", y_test.shape)

assert X_test.shape == (2947, 128, 3)
assert len(y_test) == 2947

# -------------------------------------------------
# 3. APPLY ORIGINAL TRAINING NORMALIZATION
# -------------------------------------------------

norm = np.load(NORM_PATH)

axis_mean = norm["mean"]
axis_std = norm["std"]

X_test_norm = (
    (X_test - axis_mean) / axis_std
).astype(np.float32)

# -------------------------------------------------
# 4. LOAD INT8 TFLITE MODEL
# -------------------------------------------------

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

assert inp["dtype"] == np.int8
assert out["dtype"] == np.int8

print("\nModel input:", inp["shape"])
print("Model output:", out["shape"])
print("Input quantization:", inp["quantization"])
print("Output quantization:", out["quantization"])

input_scale, input_zero = inp["quantization"]

assert input_scale > 0
assert tuple(inp["shape"]) == (1, 128, 3)

# -------------------------------------------------
# 5. GENERATE EXACT INT8 INPUTS
# Same quantization method as original notebook.
# -------------------------------------------------

X_quantized = np.round(
    X_test_norm / input_scale + input_zero
)

X_quantized = np.clip(
    X_quantized, -128, 127
).astype(np.int8)

assert X_quantized.shape == (2947, 128, 3)

# Export raw inputs for ESP32 comparison
input_path = RESULTS_DIR / "test_inputs_int8.bin"

X_quantized.tofile(input_path)

# Export ground-truth labels too
labels_path = RESULTS_DIR / "test_labels.bin"

y_test.astype(np.int8).tofile(labels_path)

# -------------------------------------------------
# 6. RUN LAPTOP INT8 INFERENCE
# Preserve RAW INT8 outputs.
# -------------------------------------------------

predictions = []
raw_outputs = []

for i in range(len(X_quantized)):

    sample = X_quantized[i:i+1]

    interpreter.set_tensor(
        inp["index"], sample
    )

    interpreter.invoke()

    raw = interpreter.get_tensor(
        out["index"]
    )[0].copy()

    raw_outputs.append(raw)

    # Argmax on raw INT8 output.
    predictions.append(int(np.argmax(raw)))

    if (i + 1) % 500 == 0:
        print(f"Processed {i+1}/2947")

raw_outputs = np.asarray(raw_outputs)
predictions = np.asarray(predictions)

assert raw_outputs.shape == (2947, 6)

# -------------------------------------------------
# 7. EXPORT LAPTOP PREDICTION CSV
# -------------------------------------------------

df = pd.DataFrame({
    "sample_index": np.arange(2947),
    "true_label": y_test,
    "prediction": predictions
})

for j in range(6):
    df[f"out_{j}"] = raw_outputs[:, j]

csv_path = RESULTS_DIR / "laptop_int8_predictions.csv"

df.to_csv(csv_path, index=False)

# -------------------------------------------------
# 8. REPORT
# -------------------------------------------------

accuracy = np.mean(predictions == y_test) * 100

print("\n========== EXPORT COMPLETE ==========")

print("Total samples:", len(y_test))
print("Input shape:", X_quantized.shape)
print("Raw output shape:", raw_outputs.shape)

print("Laptop INT8 accuracy:", round(accuracy, 4), "%")

print("\nGenerated files:")

for path in [
    input_path,
    labels_path,
    csv_path
]:
    print(path)
    print("Size:", path.stat().st_size, "bytes")

print("\n======================================")
