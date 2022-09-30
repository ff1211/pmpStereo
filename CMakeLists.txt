cmake_minimum_required(VERSION 3.9)
project(pmpStereo VERSION 1.0 LANGUAGES CXX)

add_library(pmpStereo SHARED
    src/stereoCalibrator.cpp
    src/phaseCalculator.cpp
    src/stereoProcessor.cpp
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set_target_properties(pmpStereo PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER api/pmpStereo.h
)

target_include_directories(pmpStereo PRIVATE src)
target_link_libraries(pmpStereo PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)