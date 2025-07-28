#pragma once

#include <gmock/gmock.h>
#include "ITriton.hpp"

/**
 * Mock implementation of ITriton for unit testing
 */
class MockTriton : public ITriton {
public:
    MOCK_METHOD(TritonModelInfo, getModelInfo, 
                (const std::string& modelName, const std::string& url, 
                 const std::vector<std::vector<int64_t>>& input_sizes), (override));
    
    MOCK_METHOD((std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>), 
                infer, (const std::vector<std::vector<uint8_t>>& input_data), (override));
    
    MOCK_METHOD(void, setInputShapes, (const std::vector<std::vector<int64_t>>& shapes), (override));
    
    MOCK_METHOD(void, setInputShape, (const std::vector<int64_t>& shape), (override));
    
    MOCK_METHOD(void, printModelInfo, (const TritonModelInfo& model_info), (const, override));
    
    MOCK_METHOD(void, createTritonClient, (), (override));
}; 