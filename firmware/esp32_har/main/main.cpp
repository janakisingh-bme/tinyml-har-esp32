
#include <stdio.h>
#include <cstdint>
#include <cstring>
#include <climits>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

static const char *TAG = "HAR_TINYML";

static constexpr int HAR_TEST_COUNT = 2947;
static constexpr int HAR_INPUT_SIZE = 384;

static constexpr size_t HAR_LABEL_OFFSET =
    static_cast<size_t>(HAR_TEST_COUNT) * HAR_INPUT_SIZE;

alignas(16) static uint8_t tensor_arena[100 * 1024];
static int8_t sample[HAR_INPUT_SIZE];

static const char *activity_names[] = {
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
};

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Full UCI HAR INT8 benchmark starting");

    ESP_LOGI(TAG, "Model size: %u bytes",
             static_cast<unsigned>(g_har_model_len));

    ESP_LOGI(TAG, "Test windows: %d", HAR_TEST_COUNT);

    const esp_partition_t *partition =
        esp_partition_find_first(
            ESP_PARTITION_TYPE_DATA,
            static_cast<esp_partition_subtype_t>(0x40),
            "testdata");

    if (partition == nullptr) {
        ESP_LOGE(TAG, "testdata partition not found");
        return;
    }

    const size_t required_size =
        HAR_LABEL_OFFSET + HAR_TEST_COUNT;

    if (partition->size < required_size) {
        ESP_LOGE(TAG, "Partition too small: %u bytes",
                 static_cast<unsigned>(partition->size));
        return;
    }

    ESP_LOGI(TAG, "Test partition size: %u bytes",
             static_cast<unsigned>(partition->size));

    const size_t free_before =
        heap_caps_get_free_size(MALLOC_CAP_8BIT);

    const tflite::Model *model =
        tflite::GetModel(g_har_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch");
        return;
    }

    static tflite::MicroMutableOpResolver<10> resolver;

    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddSqueeze() != kTfLiteOk ||
        resolver.AddExpandDims() != kTfLiteOk ||
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddDequantize() != kTfLiteOk) {

        ESP_LOGE(TAG, "Operator registration failed");
        return;
    }

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return;
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    if (input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8 ||
        input->bytes != HAR_INPUT_SIZE ||
        output->bytes != 6) {

        ESP_LOGE(TAG, "Unexpected input/output tensor format");
        return;
    }

    // ==========================================
    // EMBEDDED RESOURCE UTILIZATION
    // ==========================================

    ESP_LOGI(TAG, "===== MEMORY UTILIZATION =====");

    ESP_LOGI(TAG, "Model size: %u bytes",
             static_cast<unsigned>(g_har_model_len));

    ESP_LOGI(TAG, "Tensor arena reserved: %u bytes",
             static_cast<unsigned>(sizeof(tensor_arena)));

    ESP_LOGI(TAG, "Tensor arena actually used: %u bytes",
             static_cast<unsigned>(interpreter.arena_used_bytes()));

    ESP_LOGI(TAG, "Free heap before interpreter allocation: %u bytes",
             static_cast<unsigned>(free_before));

    ESP_LOGI(TAG, "Free heap after allocation: %u bytes",
             static_cast<unsigned>(
                 heap_caps_get_free_size(MALLOC_CAP_8BIT)));

    ESP_LOGI(TAG, "Minimum free heap: %u bytes",
             static_cast<unsigned>(
                 heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT)));

    ESP_LOGI(TAG, "Largest free heap block: %u bytes",
             static_cast<unsigned>(
                 heap_caps_get_largest_free_block(MALLOC_CAP_8BIT)));

    // ==========================================
    // WARM-UP INFERENCE
    // ==========================================

    for (int i = 0; i < 5; ++i) {

        if (esp_partition_read(
                partition,
                static_cast<size_t>(i) * HAR_INPUT_SIZE,
                sample, HAR_INPUT_SIZE) != ESP_OK) {

            ESP_LOGE(TAG, "Warm-up data read failed");
            return;
        }

        memcpy(input->data.int8, sample, HAR_INPUT_SIZE);

        if (interpreter.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Warm-up inference failed");
            return;
        }
    }

    // ==========================================
    // BENCHMARK INITIALIZATION
    // ==========================================

    int correct = 0;
    int class_correct[6] = {};
    int class_total[6] = {};

    int64_t total_us = 0;
    int64_t min_us = INT64_MAX;
    int64_t max_us = 0;

    printf("\nPREDICTION_LOG_START\n");

    printf("PRED,sample,true,pred,out0,out1,out2,out3,out4,out5\n");

    // ==========================================
    // FULL TEST BENCHMARK
    // ==========================================

    for (int i = 0; i < HAR_TEST_COUNT; ++i) {

        const size_t input_offset =
            static_cast<size_t>(i) * HAR_INPUT_SIZE;

        if (esp_partition_read(
                partition,
                input_offset,
                sample,
                HAR_INPUT_SIZE) != ESP_OK) {

            ESP_LOGE(TAG, "Input read failed at sample %d", i);
            return;
        }

        uint8_t label = 0;

        const size_t label_offset = HAR_LABEL_OFFSET + i;

        if (esp_partition_read(
                partition,
                label_offset,
                &label,
                sizeof(label)) != ESP_OK) {

            ESP_LOGE(TAG, "Label read failed at sample %d", i);
            return;
        }

        if (label >= 6) {
            ESP_LOGE(TAG, "Invalid label %u at sample %d",
                     static_cast<unsigned>(label), i);
            return;
        }

        memcpy(input->data.int8, sample, HAR_INPUT_SIZE);

        const int64_t start = esp_timer_get_time();

        const TfLiteStatus status = interpreter.Invoke();

        const int64_t elapsed =
            esp_timer_get_time() - start;

        if (status != kTfLiteOk) {
            ESP_LOGE(TAG, "Inference failed at sample %d", i);
            return;
        }

        int predicted = 0;

        for (int c = 1; c < 6; ++c) {

            if (output->data.int8[c] >
                output->data.int8[predicted]) {

                predicted = c;
            }
        }

        // Individual predictions and INT8 outputs.

        printf("PRED,%d,%u,%d,%d,%d,%d,%d,%d,%d\n",
               i,
               static_cast<unsigned>(label),
               predicted,
               static_cast<int>(output->data.int8[0]),
               static_cast<int>(output->data.int8[1]),
               static_cast<int>(output->data.int8[2]),
               static_cast<int>(output->data.int8[3]),
               static_cast<int>(output->data.int8[4]),
               static_cast<int>(output->data.int8[5]));

        total_us += elapsed;

        if (elapsed < min_us) min_us = elapsed;
        if (elapsed > max_us) max_us = elapsed;

        class_total[label]++;

        if (predicted == label) {
            correct++;
            class_correct[label]++;
        }

        if ((i + 1) % 250 == 0 ||
            i == HAR_TEST_COUNT - 1) {

            ESP_LOGI(TAG, "Progress: %d/%d",
                     i + 1, HAR_TEST_COUNT);
        }

        // Allow the FreeRTOS idle task to run.
        vTaskDelay(1);
    }

    printf("PREDICTION_LOG_END\n");

    // ==========================================
    // FINAL BENCHMARK SUMMARY
    // ==========================================

    ESP_LOGI(TAG, "===== FULL BENCHMARK SUMMARY =====");

    ESP_LOGI(TAG, "Correct vs labels: %d/%d (%.2f%%)",
             correct,
             HAR_TEST_COUNT,
             100.0f * correct / HAR_TEST_COUNT);

    ESP_LOGI(TAG, "Mean inference: %.2f us",
             static_cast<double>(total_us) / HAR_TEST_COUNT);

    ESP_LOGI(TAG, "Min inference: %lld us",
             (long long)min_us);

    ESP_LOGI(TAG, "Max inference: %lld us",
             (long long)max_us);

    for (int c = 0; c < 6; ++c) {

        ESP_LOGI(TAG, "%s accuracy: %d/%d (%.2f%%)",
                 activity_names[c],
                 class_correct[c],
                 class_total[c],
                 class_total[c] ?
                     100.0f * class_correct[c] / class_total[c] :
                     0.0f);
    }

    // ==========================================
    // FINAL MEMORY AND STACK MEASUREMENTS
    // ==========================================

    ESP_LOGI(TAG, "===== FINAL RESOURCE UTILIZATION =====");

    ESP_LOGI(TAG, "Tensor arena used: %u bytes",
             static_cast<unsigned>(interpreter.arena_used_bytes()));

    ESP_LOGI(TAG, "Free heap at end: %u bytes",
             static_cast<unsigned>(
                 heap_caps_get_free_size(MALLOC_CAP_8BIT)));

    ESP_LOGI(TAG, "Minimum free heap: %u bytes",
             static_cast<unsigned>(
                 heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT)));

    ESP_LOGI(TAG, "Largest free heap block: %u bytes",
             static_cast<unsigned>(
                 heap_caps_get_largest_free_block(MALLOC_CAP_8BIT)));

    // ESP-IDF reports this high-water mark in bytes.
    const UBaseType_t stack_remaining =
        uxTaskGetStackHighWaterMark(NULL);

    ESP_LOGI(TAG, "Main task stack minimum free: %u bytes",
             static_cast<unsigned>(stack_remaining));
}
