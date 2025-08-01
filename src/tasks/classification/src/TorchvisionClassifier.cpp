#include "TorchvisionClassifier.hpp"
#include "Logger.hpp"

TorchvisionClassifier::TorchvisionClassifier(const TritonModelInfo& modelInfo)
    : TaskInterface(modelInfo) {
}

std::vector<uint8_t> TorchvisionClassifier::preprocess_image(const cv::Mat& img, const std::string& format, 
                                                       int img_type1, int img_type3, size_t img_channels, 
                                                       const cv::Size& img_size) 
{
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
    sample.convertTo(sample, (img_channels == 3) ? img_type3 : img_type1);
    sample.convertTo(sample, CV_32FC3, 1.f / 255.f);
    cv::subtract(sample, cv::Scalar(0.485f, 0.456f, 0.406f), sample, cv::noArray(), -1);
    cv::divide(sample, cv::Scalar(0.229f, 0.224f, 0.225f), sample, 1, -1);
    cv::resize(sample, sample, cv::Size(img_size.width, img_size.height));

    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample.total() * sample.elemSize();
    size_t pos = 0;
    input_data.resize(img_byte_size);

    // For CHW format: BBBB...GGGG...RRRR
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i) {
        input_bgr_channels.emplace_back(img_size.height, img_size.width, img_type1, &(input_data[pos]));
        pos += input_bgr_channels.back().total() * input_bgr_channels.back().elemSize();
    }

    cv::split(sample, input_bgr_channels);

    if (pos != img_byte_size) {
        logger.errorf("Unexpected total size of channels {}, expecting {}", pos, img_byte_size);
        exit(1);
    }

    return input_data;
}

std::vector<std::vector<uint8_t>> TorchvisionClassifier::preprocess(const std::vector<cv::Mat>& imgs)
{
    if (imgs.empty()) {
        throw std::runtime_error("Input image vector is empty");
    }

    cv::Mat img = imgs.front();
    if (img.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    std::vector<std::vector<uint8_t>> input_data(model_info_.input_shapes.size());

    for (size_t i = 0; i < model_info_.input_shapes.size(); ++i) {
        const auto& input_name = model_info_.input_names[i];
        const auto& input_shape = model_info_.input_shapes[i];
        const auto& input_format = model_info_.input_formats[i];
        const auto& input_type = model_info_.input_types[i];

        if (input_shape.size() >= 3) {
            // This is likely an image input
            const auto input_size = cv::Size(input_width_, input_height_);
            input_data[i] = preprocess_image(img, input_format, model_info_.type1_, model_info_.type3_, img.channels(), input_size);
        } else {
            // For other types of inputs, you might need to add more cases
            // or use a default handling method
            throw std::runtime_error("Unhandled input");
        }
    }
    return input_data;
}

std::vector<Result> TorchvisionClassifier::postprocess(const cv::Size& frame_size, 
                                                       const std::vector<std::vector<TensorElement>>& infer_results,
                                                       const std::vector<std::vector<int64_t>>& infer_shapes) 
{
    if (infer_results.empty() || infer_shapes.empty()) {
        throw std::runtime_error("Inference results or shapes are empty.");
    }

    const auto& input_result = infer_results.front();
    const auto shape = infer_shapes[0][1];
    
    // Convert TensorElement to float
    std::vector<float> result;
    result.reserve(input_result.size());
    for (const auto& elem : input_result) {
        result.push_back(std::visit([](auto&& arg) -> float { return static_cast<float>(arg); }, elem));
    }

    // Apply exponential function
    std::transform(result.begin(), result.end(), result.begin(), 
                   [](float val) { return std::exp(val); });

    auto sum = std::accumulate(result.begin(), result.end(), 0.0f);

    // Find top classes predicted by the model
    std::vector<int> indices(shape);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&result](int i1, int i2) { return result[i1] > result[i2]; });

    // Generate results
    std::vector<Result> results;
    for (size_t i = 0; i < shape; ++i) {
        float confidence = result[indices[i]] / sum;
        if (confidence <= 0.005f) break;

        Classification classification;
        classification.class_id = indices[i];
        classification.class_confidence = confidence;
        results.emplace_back(classification);
    }

    return results;
}