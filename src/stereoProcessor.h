#ifndef STEREO_PROCESSOR
#define STEREO_PROCESSOR

#include <string>
#include <fstream>
#include "setting.h"

// Struct to initialize stereo calculator.
struct stereoConfig
{
    // File path that calibration result stored.
    std::string filePath;
    // Image Size.
    cv::Size imgSize;

    // Disparity threshold.
    TYPE disparityTH;
    // Phase match threshold.
    TYPE matchTH;

    // Search area. They must have same size.
    cv::Rect ROI1;
    cv::Rect ROI2;

    stereoConfig(cv::Size img_size, TYPE disparity_th, TYPE match_th, std::string file_path = "");
};

class stereoProcessor
{
protected:
    // Rectification parameters are calculated by stereoCalibrator using calibration results.
    cv::Mat R1;    // 3x3 rectification transform (rotation matrix) for the first camera.
    cv::Mat P1;    // 3x4 projection matrix in the new (rectified) coordinate systems for the first camera
    cv::Mat R2;    // 3x3 rectification transform (rotation matrix) for the second camera.
    cv::Mat P2;    // 3x4 projection matrix in the new (rectified) coordinate systems for the second camera
    cv::Mat map11; // Map1 for the first camera.
    cv::Mat map12; // Map2 for the first camera.
    cv::Mat map21; // Map1 for the second camera.
    cv::Mat map22; // Map2 for the second camera.

    // Disparity threshold.
    int disparityTH;
    // Phase match threshold. When left phase - right phase < matchTH, regarded as a match.
    TYPE matchTH;
    // Search area. Match points from camera1 [p1[0], p1[1]) and camera2 [p2[0], p2[1]).
    cv::Point p1[2];
    cv::Point p2[2];

    cv::Rect ROI1;
    cv::Rect ROI2;

    // Update config.
    void updateConfig(const stereoConfig &cfg);

    // Search corresponding point.
    TYPE searchPhase(TYPE x, const cv::Mat &seq, bool interpolation = true);
    //
    TYPE interpolate(TYPE x0, TYPE x1, TYPE y0, TYPE y1, TYPE x);

public:
    // Calibration results to load.
    cv::Size imgSize; // Size of phase images.
    cv::Mat K1;       // Camera matrix of the first camera.
    cv::Mat D1;       // DistCoeffs of the first camera.
    cv::Mat K2;       // Camera matrix of the second camera.
    cv::Mat D2;       // DistCoeffs of the second camera.
    cv::Mat R;        // The rotation matrix maps Camera1's coordinates to Camera2's.
    cv::Mat T;        // The translation vector maps Camera1's coordinates to Camera2's.
    cv::Mat F;        // Fundamental matrix.
    cv::Mat E;        // Essential matrix.
    cv::Mat Q;        // Disparity-to-depth mapping matrix.

    // Constructor.
    stereoProcessor(const stereoConfig &cfg);
    // Destructor.
    ~stereoProcessor();
    // Calculate rectify map.
    void calRectifyMap();
    // Load calibration result saved in xml file.
    void loadCaliResult(std::string filePath);
    // Remap to get rectified image.
    void rectifyRemap(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst1, cv::Mat &dst2);
    // Match phase map and calculate disparity.
    void calDisparity(const cv::Mat &absPhase1, const cv::Mat &absPhase2, cv::Mat &disparity, bool interpolation = true);
};

#endif