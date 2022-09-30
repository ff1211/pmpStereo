#include "stereoProcessor.h"

using namespace std;
using namespace cv;

stereoProcessor::stereoProcessor(const stereoConfig &cfg)
{
    updateConfig(cfg);
    calRectifyMap();
}

stereoProcessor::~stereoProcessor() {}

void stereoProcessor::updateConfig(const stereoConfig &cfg)
{
    imgSize = cfg.imgSize;
    matchTH = cfg.matchTH;
    disparityTH = cfg.disparityTH;
    ROI1 = cfg.ROI1;
    ROI2 = cfg.ROI2;

    if (cfg.filePath != "")
        loadCaliResult(cfg.filePath);
}

void stereoProcessor::loadCaliResult(string filePath)
{
    // Create matrix.
    K1.create(3, 3, CV_64FC1);
    D1.create(1, 5, CV_64FC1);
    K2.create(3, 3, CV_64FC1);
    D2.create(1, 5, CV_64FC1);
    R.create(3, 3, CV_64FC1);
    T.create(3, 1, CV_64FC1);
    F.create(3, 3, CV_64FC1);
    E.create(3, 3, CV_64FC1);
    R1.create(3, 3, CV_64FC1);
    P1.create(3, 4, CV_64FC1);
    R2.create(3, 3, CV_64FC1);
    P2.create(3, 4, CV_64FC1);
    Q.create(4, 4, CV_64FC1);

    // Open calibration file and load parameters.
    ifstream caliResult;
    caliResult.open(filePath, ios::in);
    // Check the file.
    if (caliResult.fail())
        throw "Can't open file!";

    // Load camera matrix of the first camera.
    for (int i = 0; i < K1.rows; ++i)
        for (int j = 0; j < K1.cols; ++j)
            caliResult >> K1.at<double>(i, j);
    // Load distCoeffs of the first camera.
    for (int j = 0; j < D1.cols; ++j)
        caliResult >> D1.at<double>(j);

    // Load camera matrix of the second camera.
    for (int i = 0; i < K2.rows; ++i)
        for (int j = 0; j < K2.cols; ++j)
            caliResult >> K2.at<double>(i, j);
    // Load distCoeffs of the second camera.
    for (int j = 0; j < D2.cols; ++j)
        caliResult >> D2.at<double>(j);

    // Load the rotation matrix maps Camera1's coordinates to Camera2's.
    for (int i = 0; i < R.rows; ++i)
        for (int j = 0; j < R.cols; ++j)
            caliResult >> R.at<double>(i, j);
    // cout << R << endl;
    // Load the translation vector maps Camera1's coordinates to Camera2's.
    for (int i = 0; i < T.rows; ++i)
        for (int j = 0; j < T.cols; ++j)
            caliResult >> T.at<double>(i, j);

    // Load fundamental matrix.
    for (int i = 0; i < F.rows; ++i)
        for (int j = 0; j < F.cols; ++j)
            caliResult >> F.at<double>(i, j);

    // Load essential matrix.
    for (int i = 0; i < E.rows; ++i)
        for (int j = 0; j < E.cols; ++j)
            caliResult >> E.at<double>(i, j);

    caliResult.close();
    calRectifyMap();
}

void stereoProcessor::rectifyRemap(const Mat &src1, const Mat &src2, Mat &dst1, Mat &dst2, bool filter)
{
    // Remap phase images.
#pragma omp parallel sections private(filter)
    {
#pragma omp section
        {
            remap(src1, dst1, map11, map12, INTER_LINEAR, BORDER_CONSTANT);
            if (filter)
                medianBlur(dst1, dst1, PHASE_FILTER_WINSIZE);
        }
#pragma omp section
        {
            remap(src2, dst2, map21, map22, INTER_LINEAR, BORDER_CONSTANT);
            if (filter)
                medianBlur(dst2, dst2, PHASE_FILTER_WINSIZE);
        }
    }
}

void stereoProcessor::calRectifyMap()
{
    // Stereo rectify.
    stereoRectify(K1, D1, K2, D2, imgSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);
    // Calculate rectification map.
    initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_16SC2, map11, map12);
    initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_16SC2, map21, map22);
}

void stereoProcessor::calDisparity(const Mat &phase1, const Mat &phase2, unsigned int windowSize, Mat &disparity, bool interpolation)
{
    if (windowSize == 0)
        throw "Window size must larger than 0!";
    // Allocate memory for disparity map.
    disparity.create(phase1.size(), CV_TYPE);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < disparity.rows; ++i)
    {
        for (int j = 0; j < disparity.cols; ++j)
        {
            if (ROI1.contains(Point(j, i)))
            {
                TYPE x = phase1.at<TYPE>(i, j);
                TYPE matchPoint = (x > 30) ? binarySearch(x, windowSize, interpolation, phase2.row(i)) : -1;
                if (matchPoint > -1)
                    disparity.at<TYPE>(i, j) = (j - matchPoint) > disparityTH ? j - matchPoint : 0;
                else
                    disparity.at<TYPE>(i, j) = 0;
            }
            else
                disparity.at<TYPE>(i, j) = 0;
        }
    }
}

TYPE stereoProcessor::binarySearch(TYPE x, unsigned int windowSize, bool interpolation, const Mat &seq)
{
    // Cache number of windows during binarySearch.
    int m = (ROI1.width + windowSize - 1) / windowSize;

    int low = 0, high = m - 1;
    int mid;

    while (high >= low)
    {
        mid = (high + low) / 2;
        TYPE wl = mid * windowSize;
        TYPE wr = mid * windowSize + windowSize - 1;

        if (x < seq.at<TYPE>(wl))
            high = mid - 1;
        else if (x > seq.at<TYPE>(wr))
            low = mid + 1;
        else
        {
            TYPE tmp0 = x - seq.at<TYPE>(wl);
            int j = wl;
            for (int i = wl + 1; i <= wr && i < seq.cols; ++i)
            {
                if (abs(x - seq.at<TYPE>(i)) < abs(tmp0))
                {
                    tmp0 = x - seq.at<TYPE>(i);
                    j = i;
                }
            }

            if (abs(tmp0) < matchTH)
                if (tmp0 > 0)
                    return (j + 1 < seq.cols) ? interpolate(j, j + 1, seq.at<TYPE>(j), seq.at<TYPE>(j + 1), x) : j;
                else
                    return (j - 1 < 0) ? j : interpolate(j - 1, j, seq.at<TYPE>(j - 1), seq.at<TYPE>(j), x);
            else
                return -1;
        }
    }
    return -1;
}

TYPE stereoProcessor::interpolate(TYPE x0, TYPE x1, TYPE y0, TYPE y1, TYPE x)
{
    if (x0 == x1)
        throw "False interpolation input!";

    return (x - x1) * y0 / (x0 - x1) + (x - x0) * y1 / (x1 - x0);
}