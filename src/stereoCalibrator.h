#ifndef STEREO_CALIBRATOR
#define STEREO_CALIBRATOR

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class stereoCalibrator
{
private:
    std::vector<cv::Point3f> objectPoints;
    std::string caliImgPath1, caliImgPath2;
    cv::Size boardSize;
    int imagePairs;
    float squareSize;

public:
    stereoCalibrator(int image_pairs, int board_width, int board_height, float square_size);
    ~stereoCalibrator();
    void updateCaliConfig(int image_pairs, int board_width, int board_height, float square_size);
    void calibrate(std::string caliImgPath1, std::string caliImgPath2, cv::Size &imgSize, cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2, cv::Mat &R, cv::Mat &T, cv::Mat &F, cv::Mat &E);
};
#endif