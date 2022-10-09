#ifndef STEREO
#define STEREO

#include <opencv2/opencv.hpp>
#include <omp.h>

// N-Step algorithm.
#define THREE_STEP_SHIFT 0
#define FOUR_STEP_SHIFT 1
// Steps to calculate final hetero phase map.
#define TWO_STEP_HETERODYNE 0
#define THREE_STEP_HETERODYNE 1
// Math quantities.
#define ROOT_THREE 1.73205081
#define TWO_THIRD 0.66666667
#define PI_2 6.28318531
// Floor precision.
#define FLOOR_PRECISION 0.5
// Type define
#define FLOAT_PRECISION
// #define DOUBLE_PRECISION

#if defined FLOAT_PRECISION
typedef float TYPE;
#define CV_TYPE CV_32FC1
#elif defined DOUBLE_PRECISION
typedef double TYPE;
#define CV_TYPE CV_64FC1
#endif

// Phase filter window size.
#define PHASE_FILTER_WINSIZE 3

#endif