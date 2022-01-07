cmake_minimum_required(VERSION 3.20)
project(22_01_07_HyperPose_Example)

include_directories(/home/rick/local_install/hyperpose/include)
link_directories(/home/rick/local_install/hyperpose/build)

include_directories(/usr/local/cuda-11.0/include)
link_directories(/usr/local/cuda-11.0/lib64)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(22_01_07_HyperPose_Example main.cpp)
target_link_libraries(22_01_07_HyperPose_Example ${OpenCV_LIBS} Threads::Threads hyperpose cudart cudnn nvinfer nvonnxparser nvparsers)
