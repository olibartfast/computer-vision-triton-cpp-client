#include "TensorflowClassifier.hpp"

TensorflowClassifier::TensorflowClassifier(int input_width, int input_height, int channels)
    : TaskInterface(input_width, input_height, channels) {
}


std::vector<uint8_t> TensorflowClassifier::preprocess(const cv::Mat& img, const std::string& format,
                                                     int img_type1, int img_type3, size_t img_channels,
                                                     const cv::Size& img_size)
{
    cv::Mat sample;
    img.convertTo(sample, (img_channels == 3) ? img_type3 : img_type1);
    cv::resize(sample, sample, cv::Size(img_size.width, img_size.height));

    // Allocate a buffer to hold all image elements
    size_t img_byte_size = sample.total() * sample.elemSize();
    size_t pos = 0;
    std::vector<uint8_t> input_data(img_byte_size);

    // Copy data maintaining NHWC format
    if (sample.isContinuous()) {
        memcpy(&(input_data[0]), sample.datastart, img_byte_size);
        pos = img_byte_size;
    } else {
        size_t row_byte_size = sample.cols * sample.elemSize();
        for (int r = 0; r < sample.rows; ++r) {
            memcpy(&(input_data[pos]), sample.ptr<uint8_t>(r), row_byte_size);
            pos += row_byte_size;
        }
    }

    return input_data;
}


std::vector<Result> TensorflowClassifier::postprocess(const cv::Size& frame_size,
                                                       const std::vector<std::vector<float>>& infer_results,
                                                       const std::vector<std::vector<int64_t>>& infer_shapes) {
    // Check if the input vectors are not empty
    if (infer_results.empty() || infer_shapes.empty()) {
        throw std::runtime_error("Inference results or shapes are empty.");
    }

    // Get the first set of results and shape
    const auto& results = infer_results.front();
    const auto shape = infer_shapes[0][1];

    // Define the confidence threshold and top-k values
    const auto confThreshold = 0.5f;
    size_t i = 0;
    std::vector<int> indices(shape);
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&results](int i1, int i2) { return results[i1] > results[i2]; });

    // Extract top-k results with confidence greater than the threshold
    float threshold = 0.5f;
    std::vector<Result> classification_results;
    while (results[indices[i]] > threshold) {
        Classification classification;
        classification.class_id = indices[i];
        classification.class_confidence = results[indices[i]];
        ++i;
        classification_results.emplace_back(classification);
    }

    return classification_results;
}

