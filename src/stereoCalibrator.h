#ifndef STEREO_CALIBRATOR
#define STEREO_CALIBRATOR

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class stereoCalibrator
{
private:
    std::vector<cv::Point3f> objectPoints;
    cv::Size boardSize;
    float squareSize;

public:
    stereoCalibrator(cv::Size _boardSize, float _squareSize);
    ~stereoCalibrator();
    void updateCaliConfig(cv::Size _boardSize, float _squareSize);
    void calibrate(const std::vector<cv::Mat> &imgSet1, const std::vector<cv::Mat> &imgSet2, cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2, cv::Mat &R, cv::Mat &T, cv::Mat &F, cv::Mat &E) const;
};
#endif