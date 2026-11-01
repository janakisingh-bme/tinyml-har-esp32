# Performance–Efficiency Trade-Offs in Quantized Neural Networks for Human Activity Recognition at the Edge

**An empirical evaluation of INT8 quantization and embedded inference using a lightweight 1D CNN on ESP32.**

## 1. Research Motivation

The rapid growth of **Edge AI and Tiny Machine Learning (TinyML)** is enabling intelligent applications to operate directly on resource-constrained devices rather than relying entirely on cloud computing. This shift is particularly relevant to wearable healthcare, human activity monitoring, and IoT systems, where low latency, reduced communication overhead, and local processing are important.

However, deploying deep learning models on microcontrollers introduces a fundamental challenge: balancing classification performance with limited computational resources.

This research investigates the **performance–efficiency trade-offs** of quantized neural networks for Human Activity Recognition (HAR), evaluating classification accuracy, model size, inference latency, and memory utilization through an ESP32 implementation.

## 2. Research Objectives

- Develop a lightweight 1D CNN for accelerometer-based human activity recognition.
- Evaluate the impact of INT8 quantization on model size and classification performance.
- Deploy the quantized model using TensorFlow Lite Micro on ESP32.
- Measure on-device inference latency and memory utilization.
- Analyze the trade-off between recognition performance and embedded resource requirements.

## 3. Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | UCI HAR |
| Sensor data | 3-axis accelerometer |
| Input shape | `(128, 3)` |
| Sampling frequency | 50 Hz |
| Window duration | 2.56 seconds |
| Activity classes | 6 |
| Model | Lightweight 1D CNN |
| Trainable parameters | 2,702 |
| Quantization | INT8 |
| Hardware | ESP32 DevKit V1 |
| Embedded framework | ESP-IDF |
| Inference framework | TensorFlow Lite Micro |

The official subject-independent training and testing partitions are retained, containing 7,352 training windows and 2,947 testing windows.

## 4. Experimental Results

### Model comparison

| Metric | Float32 TFLite | INT8 TFLite |
|---|---:|---:|
| Model size | 16,980 B | 11,328 B |
| Accuracy | 85.58% | 85.27% |
| Macro-F1 | 85.64% | 85.34% |

**Key finding:** INT8 quantization reduces the recorded model artifact size by approximately **33.3%**, with an accuracy reduction of approximately **0.31 percentage points**.

### ESP32 deployment results

| Metric | Result |
|---|---:|
| On-device accuracy | **85.61%** |
| Mean inference latency | **18.43 ms/window** |
| Tensor arena used | 6,508 bytes |
| Minimum free heap after interpreter initialization | 198,868 bytes |
| Model parameters | 2,702 |

### Class-wise accuracy

| Activity | Accuracy |
|---|---:|
| Walking | 93.15% |
| Walking Upstairs | 80.04% |
| Walking Downstairs | 90.24% |
| Sitting | 78.41% |
| Standing | 71.99% |
| Laying | 100.00% |

## 5. Research Contribution

This work provides an empirical evaluation of the trade-offs involved in deploying quantized neural networks on resource-constrained edge hardware.

The key contributions are:

- Development of a lightweight CNN for multivariate time-series classification.
- Experimental comparison of Float32 and INT8 TFLite model artifacts.
- Successful deployment of quantized inference on an ESP32 microcontroller.
- Measurement of embedded inference latency and memory requirements.
- Analysis of class-wise recognition performance and deployment limitations.

## 6. Limitations and Future Work

The current ESP32 implementation performs offline inference using preprocessed test windows stored in flash. Live sensor acquisition and continuous real-time activity recognition have not yet been implemented. Power consumption has also not been measured.

Future work will focus on:

- Real-time accelerometer integration.
- Energy and power consumption measurement.
- Further model optimization and architecture comparison.
- Evaluation across additional embedded platforms.
- Robustness testing under real-world activity conditions.

## 7. Technologies

**Python · TensorFlow/Keras · TensorFlow Lite · TensorFlow Lite Micro · ESP-IDF · C++ · ESP32 · NumPy · Scikit-learn**

## 8. References

1. D. Anguita et al., “A Public Domain Dataset for Human Activity Recognition Using Smartphones,” ESANN, 2013.
2. UCI Machine Learning Repository: [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
3. TensorFlow: [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers).
4. Espressif Systems: [ESP-IDF Documentation](https://docs.espressif.com/projects/esp-idf/en/stable/).

---

**Research Focus:** TinyML · Edge AI · Quantized Neural Networks · Human Activity Recognition · Embedded Intelligence

*Academic research and engineering implementation. Not intended for medical diagnosis.*

**Note:** Float32 and INT8 figures are recorded TFLite artifact results; a common training-checkpoint provenance has not been verified. ESP32 accuracy and latency refer to the separate on-device offline evaluation.