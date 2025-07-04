# Use an official Nvidia Triton Inference Server image as the base image
# docker build --rm -t computer-vision-triton-cpp-client .
ARG TRITON_VERSION=25.06
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3-sdk

# Install any additional dependencies if needed
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y build-essential cmake
RUN apt-get install -y libcurl4-openssl-dev
RUN apt-get install -y rapidjson-dev libprotobuf-dev protobuf-compiler 

# Set environment variables
ENV TritonClientBuild_DIR /workspace/install


# Copy your C++ source code and CMakeLists.txt into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Build your C++ application
RUN rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build .

# Set the entry point for the container
ENTRYPOINT ["/app/build/computer-vision-triton-cpp-client"]

# Default command if no arguments are provided
CMD ["--help"]
