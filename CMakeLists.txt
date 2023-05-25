cmake_minimum_required(VERSION 2.6)

project(srcnn)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/home/nvidia/TensorRT-8.2.5.1/include/)
link_directories(/home/nvidia/TensorRT-8.2.5.1/lib/)

# include and link dirs of cuda for inference
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)

# opencv library
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV library status:")
message(STATUS "version:${OpenCV_VERSION}")
message(STATUS "libraries:${OpenCV_LIBS}")
message(STATUS "include path:${OpenCV_INCLUDE_DIRS}")

# link library and add exec file
add_executable(srcnn ${PROJECT_SOURCE_DIR}/srcnn.cpp)
target_link_libraries(srcnn nvinfer)
target_link_libraries(srcnn cudart)
target_link_libraries(srcnn ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

