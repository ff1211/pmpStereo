#include "stereoProcessor.h"

using namespace std;
using namespace cv;

stereoConfig::stereoConfig(cv::Size img_size, TYPE disparity_th, TYPE match_th, std::string file_path) : imgSize(img_size),
                                                                                                         disparityTH(disparity_th),
                                                                                                         matchTH(match_th),
                                                                                                         filePath(file_path)
{
}

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
    {
        cout << "Can't open calibration file!" << endl;
        throw exception();
    }

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

void stereoProcessor::rectifyRemap(const Mat &src1, const Mat &src2, Mat &dst1, Mat &dst2)
{
    // Remap phase images.
    remap(src1, dst1, map11, map12, INTER_LINEAR, BORDER_CONSTANT);
    remap(src2, dst2, map21, map22, INTER_LINEAR, BORDER_CONSTANT);
}

void stereoProcessor::calRectifyMap()
{
    // Stereo rectify.
    Rect validPixROI1, validPixROI2;
    stereoRectify(K1, D1, K2, D2, imgSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imgSize, &validPixROI1, &validPixROI2);
    ROI1.x = validPixROI1.x;
    ROI1.width = validPixROI1.width;
    ROI2.x = validPixROI2.x;
    ROI2.width = validPixROI2.width;

    if (validPixROI1.y > validPixROI2.y)
        ROI1.y = validPixROI1.y;
    else
        ROI1.y = validPixROI2.y;

    if (validPixROI1.br().y < validPixROI2.br().y)
        ROI1.height = validPixROI1.br().y - ROI1.y;
    else
        ROI1.height = validPixROI2.br().y - ROI1.y;

    ROI2.y = ROI1.y;
    ROI2.height = ROI1.height;

    // Calculate rectification map.
    initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_16SC2, map11, map12);
    initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_16SC2, map21, map22);
}

void stereoProcessor::calDisparity(const Mat &absPhase1, const Mat &absPhase2, Mat &disparity, bool interpolation)
{
    // Allocate memory for disparity map.
    disparity.create(absPhase1.size(), CV_TYPE);
    Mat absPhase2ROI = absPhase2(ROI2);
#pragma omp parallel for
    for (int i = 0; i < disparity.rows; ++i)
    {
        for (int j = 0; j < disparity.cols; ++j)
        {
            if (ROI1.contains(Point(j, i)))
            {
                TYPE x = absPhase1.at<TYPE>(i, j);

                TYPE matchPoint = isnan(x)? -1 : searchPhase(x, absPhase2ROI.row(i - ROI2.y), interpolation) + ROI2.x;
                if (matchPoint > -1)
                    disparity.at<TYPE>(i, j) = (j - matchPoint) > disparityTH ? j - matchPoint : 0;
                else
                    disparity.at<TYPE>(i, j) = NAN;
            }
            else
                disparity.at<TYPE>(i, j) = NAN;
        }
    }
}

TYPE stereoProcessor::searchPhase(TYPE x, const Mat &seq, bool interpolation)
{
    TYPE delta = matchTH;
    int j = 0;
    for (int i = 0; i < seq.cols; i++)
    {
        if (~isnan(seq.at<TYPE>(i)))
        {
            TYPE tmp = abs(x - seq.at<TYPE>(i));
            if (tmp < delta)
            {
                delta = tmp;
                j = i;
            }
        }
    }
    TYPE ans;
    if (delta < matchTH)
    {
        if(x - seq.at<TYPE>(j) > 0)
            return (j == seq.cols - 1)? j : interpolate(seq.at<TYPE>(j), seq.at<TYPE>(j+1), j, j+1, x);
        else
            return (j == 0)? j : interpolate(seq.at<TYPE>(j-1), seq.at<TYPE>(j), j-1, j, x);
    }
    else
        return -1;
}

TYPE stereoProcessor::interpolate(TYPE x0, TYPE x1, TYPE y0, TYPE y1, TYPE x)
{
    if (x0 == x1)
    {
        cout << "False interpolation input!" << endl;
        throw exception();
    }

    return (x - x1) * y0 / (x0 - x1) + (x - x0) * y1 / (x1 - x0);
}