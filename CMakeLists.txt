cmake_minimum_required(VERSION 3.9)
project(stereo  VERSION 1.0 LANGUAGES CXX)

add_executable( stereo main.cpp stereoProcessor.cpp phaseCalculator.cpp stereoCalibrator.cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED )

target_link_libraries( stereo PUBLIC ${OpenCV_LIBS}  OpenMP::OpenMP_CXX)