#include "stereoCalibrator.h"

using namespace std;
using namespace cv;

stereoCalibrator::stereoCalibrator(cv::Size _boardSize, float _squareSize)
{
    updateCaliConfig(_boardSize, _squareSize);
}

stereoCalibrator::~stereoCalibrator()
{
}

void stereoCalibrator::updateCaliConfig(cv::Size _boardSize, float _squareSize)
{
    // Update chessboard parameters.
    boardSize = _boardSize;
    squareSize = _squareSize;

    // Calculate chessboard coordinates.
    for (int i = 0; i < boardSize.height; i++)
        for (int j = 0; j < boardSize.width; j++)
            objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));
}

void stereoCalibrator::calibrate(const std::vector<cv::Mat> &imgSet1, const std::vector<cv::Mat> &imgSet2, cv::Mat &K1, cv::Mat &D1, cv::Mat &K2, cv::Mat &D2, cv::Mat &R, cv::Mat &T, cv::Mat &F, cv::Mat &E) const
{
    if(imgSet1.empty() | imgSet2.empty())
    {
        cout << "None calibration images!" << endl;
        throw exception();
    }
    else if(imgSet1.size() != imgSet2.size())
    {
        cout << "Different number of calibration images!" << endl;
        throw exception();
    }

    // Find corners.
    bool found1, found2;
    vector<vector<Point3f>> objectPointsSet;
    vector<vector<Point2f>> imgPoints1, imgPoints2;
    for (int i = 0; i < imgSet1.size(); i++)
    {
        vector<Point2f> cornerPoints1, cornerPoints2;
        // Finding checker board corners
        // If desired number of corners are found in the image then found = true
        found1 = findChessboardCorners(imgSet1[i], boardSize, cornerPoints1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
        found2 = findChessboardCorners(imgSet2[i], boardSize, cornerPoints2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

        if (found1 & found2)
        {
            TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
            // Refining pixel coordinates for given 2d points.
            cornerSubPix(imgSet1[i], cornerPoints1, Size(11, 11), Size(-1, -1), criteria);
            cornerSubPix(imgSet2[i], cornerPoints2, Size(11, 11), Size(-1, -1), criteria);

            objectPointsSet.push_back(objectPoints);

            imgPoints1.push_back(cornerPoints1);
            imgPoints2.push_back(cornerPoints2);
        }
    }
    stereoCalibrate(objectPointsSet, imgPoints1, imgPoints2, K1, D1, K2, D2, imgSet1[0].size(), R, T, E, F);
}