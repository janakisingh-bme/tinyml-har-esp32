
# TinyML Human Activity Recognition on ESP32

A resource-constrained TinyML research project that classifies
six human activities using accelerometer data and a lightweight
1D convolutional neural network deployed on an ESP32.

## Activities
- Walking
- Walking upstairs
- Walking downstairs
- Sitting
- Standing
- Laying

## Dataset
UCI Human Activity Recognition Using Smartphones (UCI HAR).

Input: 128 time steps x 3 accelerometer axes.
Sampling frequency: 50 Hz.
Window duration: 2.56 seconds.

## Models
- Random Forest baseline
- Float32 1D CNN
- Fully quantized INT8 1D CNN

## Evaluation
- Test accuracy and macro-F1
- Confusion matrix
- Model file size
- ESP32 inference latency
- Runtime memory usage
- Agreement between laptop and ESP32 predictions

## Status
- [ ] Repository and environment setup
- [ ] Dataset loading and exploration
- [ ] Random Forest baseline
- [ ] Train lightweight CNN
- [ ] Quantize to INT8
- [ ] Deploy to ESP32
- [ ] Benchmark and document results
