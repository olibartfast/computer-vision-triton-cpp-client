#include "task_factory.hpp"
#include "DFine.hpp"
#include "RTDetr.hpp"
#include "RTDetrUltralytics.hpp"
#include "YOLO.hpp"
#include "YoloNas.hpp"
#include "YOLOv10.hpp"
#include "TorchvisionClassifier.hpp"
#include "TensorflowClassifier.hpp"
#include "YOLOSeg.hpp"
#include "RAFT.hpp"
// Define the map of task creators
std::map<std::string, TaskFactory::TaskCreator> TaskFactory::taskCreators = {
    {"torchvision-classifier", [](const TritonModelInfo& modelInfo) { return std::make_unique<TorchvisionClassifier>(modelInfo); }},
    {"tensorflow-classifier", [](const TritonModelInfo& modelInfo) { return std::make_unique<TensorflowClassifier>(modelInfo); }},
    {"yoloseg", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLOSeg>(modelInfo); }},
    {"yolonas", [](const TritonModelInfo& modelInfo) { return std::make_unique<YoloNas>(modelInfo); }},
    {"yolov5", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov6", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov7", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov8", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov9", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolo11", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLO>(modelInfo); }},
    {"yolov10", [](const TritonModelInfo& modelInfo) { return std::make_unique<YOLOv10>(modelInfo); }},
    {"rtdetr", [](const TritonModelInfo& modelInfo) { return std::make_unique<RTDetr>(modelInfo); }},
    {"rtdetrv2", [](const TritonModelInfo& modelInfo) { return std::make_unique<RTDetr>(modelInfo); }},
    {"rtdetrul", [](const TritonModelInfo& modelInfo) { return std::make_unique<RTDetrUltralytics>(modelInfo); }},
    {"dfine", [](const TritonModelInfo& modelInfo) { return std::make_unique<DFine>(modelInfo); }},
    {"raft",  [](const TritonModelInfo& modelInfo) { return std::make_unique<RAFT>(modelInfo); }}
};

void TaskFactory::validateInputSizes(const std::vector<std::vector<int64_t>>& input_sizes) {
    if (input_sizes.empty()) {
        throw std::invalid_argument("Input sizes vector is empty");
    }
    for (const auto& size : input_sizes) {
        if (size.empty()) {
            throw std::invalid_argument("An input size vector is empty");
        }
        if (std::any_of(size.begin(), size.end(), [](int64_t s) { return s < 0; })) {
            throw std::invalid_argument("Negative input size detected");
        }
    }
}

std::unique_ptr<TaskInterface> TaskFactory::createTaskInstance(const std::string& modelType, const TritonModelInfo& modelInfo) {
    try {
        const auto& input_sizes = modelInfo.input_shapes;
        validateInputSizes(input_sizes);

        auto it = taskCreators.find(modelType);
        if (it != taskCreators.end()) {
            return it->second(modelInfo);
        }

        throw std::runtime_error("Unrecognized model type: " + modelType);
    }
    catch (const std::exception& e) {
        std::cerr << "Error creating task instance: " << e.what() << std::endl;
        throw; // Re-throw the exception
    }
}