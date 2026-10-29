# TinyML Human Activity Recognition on ESP32

A resource-efficient human activity recognition (HAR) system using a lightweight 1D Convolutional Neural Network (CNN), TensorFlow Lite quantization, and an ESP32 microcontroller.

The project investigates the trade-off between **classification performance, model size, and embedded inference efficiency** for on-device human activity recognition.

## Project Overview

Human activity recognition can support wearable devices, fitness monitoring, assistive technologies, and context-aware IoT applications. Deploying machine learning on microcontrollers introduces constraints on flash storage, RAM, and inference latency.

This project develops and evaluates a lightweight CNN using accelerometer data from the UCI Human Activity Recognition dataset, converts the model to TensorFlow Lite, and prepares it for ESP32 deployment using ESP-IDF.

### Objectives

- Develop a lightweight CNN for six-class human activity recognition.
- Evaluate a feature-based machine learning baseline against a 1D CNN.
- Compare float32 and INT8-quantized model size and classification performance.
- Integrate the quantized model into an ESP32 firmware project.
- Measure embedded inference latency and memory usage on the target hardware.

## Dataset

**Dataset:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

The project uses the three accelerometer channels:

- `total_acc_x`
- `total_acc_y`
- `total_acc_z`

| Property | Value |
|---|---|
| Training windows | 7,352 |
| Test windows | 2,947 |
| Window length | 128 samples |
| Sensor channels | 3 |
| Sampling frequency | 50 Hz |
| Window duration | 2.56 seconds |
| Input shape | `(128, 3)` |
| Number of classes | 6 |

### Activity Classes

1. Walking
2. Walking upstairs
3. Walking downstairs
4. Sitting
5. Standing
6. Laying

The dataset provides predefined training and test partitions with different subjects in each partition. The official split is retained to evaluate generalization to unseen subjects.

## Methodology

The project follows this workflow:

1. Load and inspect the UCI HAR dataset.
2. Prepare three-channel accelerometer windows.
3. Train and evaluate a feature-based baseline.
4. Train a lightweight 1D CNN.
5. Convert the CNN to TensorFlow Lite.
6. Apply INT8 quantization and evaluate the quantized model.
7. Integrate the model into ESP32 firmware using ESP-IDF.
8. Benchmark inference latency and memory usage on the microcontroller.

## Model Architecture

The primary deep-learning model is a lightweight **1D CNN** designed for sequential sensor data.

Each input window contains 128 time steps and three accelerometer channels. The model predicts one of the six activity classes.

The CNN contains approximately **2,702 trainable parameters**, making it suitable for investigating resource-constrained inference.

## Experimental Results

### Classification Performance

The following results were obtained on the held-out test set.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Feature-based baseline | 79.91% | 0.7983 |
| Float32 1D CNN | 85.58% | 0.8564 |
| INT8 TFLite CNN | 85.27% | 0.8534 |

The CNN improves classification performance over the feature-based baseline. INT8 quantization reduces the model's storage footprint while retaining performance close to the float32 model.

### Model Size and Laptop Inference

| Metric | Float32 TFLite | INT8 TFLite |
|---|---:|---:|
| Model size | 16,980 bytes | 11,328 bytes |
| Model size (approx.) | 16.58 KiB | 11.06 KiB |
| Test accuracy | 85.58% | 85.27% |
| Macro-F1 | 0.8564 | 0.8534 |
| ESP32 inference latency | Pending measurement | Pending measurement |
| ESP32 RAM usage | Pending measurement | Pending measurement |

INT8 quantization reduces the model file size by approximately **33.3%**, with an accuracy decrease of approximately **0.31 percentage points**.

Laptop inference timings are not presented as ESP32 performance measurements. Embedded latency and memory usage must be measured on the actual target board.

## Hardware and Software

### Hardware

- ESP32 DevKit V1

### Software

- Python
- NumPy
- scikit-learn
- TensorFlow
- TensorFlow Lite
- ESP-IDF
- C/C++

## Repository Structure

```text
tinyml-har-esp32/
├── docs/
│   └── experiments.md
├── firmware/
│   └── esp32_har/
│       ├── main/
│       ├── model_data.h
│       ├── test_data.h
│       ├── CMakeLists.txt
│       └── idf_component.yml
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── 02_train_cnn.ipynb
│   └── 03_quantize_cnn.ipynb
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   └── export_test_data.py
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

*The tree is illustrative. Keep only paths that exist in your current repository.*

## Running the Python Experiments

Clone the repository:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd tinyml-har-esp32
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the exploratory analysis and training/quantization notebooks from the `notebooks/` directory using Jupyter.

## ESP32 Deployment

The firmware project is located in `firmware/esp32_har/` and is intended for ESP-IDF.

The deployment workflow is:

1. Configure the ESP-IDF environment.
2. Build the firmware for the target ESP32.
3. Flash the firmware to the board.
4. Run inference using the integrated model.
5. Record inference latency, model storage, and runtime memory usage.

**Embedded benchmark results will be added after validation on the physical ESP32 board.** Laptop measurements are not substitutes for MCU measurements.

## Current Status

- [x] Dataset preparation and exploratory analysis
- [x] Feature-based baseline evaluation
- [x] Lightweight 1D CNN training
- [x] Float32 TFLite conversion
- [x] INT8 quantization and evaluation
- [ ] Complete ESP32 firmware build and deployment
- [ ] Measure ESP32 inference latency and RAM usage
- [ ] Finalize reproducible embedded benchmarks

## Future Work

- Benchmark inference latency across repeated runs on the ESP32.
- Measure peak runtime RAM and flash usage.
- Evaluate the impact of quantization on embedded performance.
- Investigate energy consumption and real-time sensor acquisition.
- Test activity recognition using live accelerometer data.

## Research Focus

This project explores the practical deployment of compact neural networks on resource-constrained embedded devices, with emphasis on the trade-offs among model accuracy, storage requirements, memory consumption, and inference latency.

