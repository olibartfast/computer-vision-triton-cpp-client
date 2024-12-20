#include "TorchvisionClassifier.hpp"

TorchvisionClassifier::TorchvisionClassifier(int input_width, int input_height, int channels)
    : TaskInterface(input_width, input_height, channels) {
}

std::vector<Result> TorchvisionClassifier::postprocess(const cv::Size& frame_size, 
                                                       const std::vector<std::vector<float>>& infer_results,
                                                       const std::vector<std::vector<int64_t>>& infer_shapes) 
{
    // Create a mutable copy of the result vector since infer_results is const
    std::vector<float> result = infer_results.front();
    const auto shape = infer_shapes[0][1];
    
    // Now transform the mutable copy
    std::transform(result.begin(), result.end(), result.begin(), 
                  [](float val) { return std::exp(val); });

    auto sum = std::accumulate(result.begin(), result.end(), 0.0);

    // find top classes predicted by the model
    std::vector<int> indices(shape);
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&result](int i1, int i2) { return result[i1] > result[i2]; });

    // print results
    size_t i = 0;
    std::vector<Result> results;
    while (result[indices[i]] / sum > 0.005) {
        Classification classification;
        classification.class_id = indices[i];
        classification.class_confidence = result[indices[i]] / sum;
        ++i;
        results.emplace_back(classification);
    }

    return results;
}

std::vector<uint8_t> TorchvisionClassifier::preprocess(const cv::Mat& img, const std::string& format, 
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
        std::cerr << "Unexpected total size of channels " << pos << ", expecting " << img_byte_size << std::endl;
        exit(1);
    }

    return input_data;
}
