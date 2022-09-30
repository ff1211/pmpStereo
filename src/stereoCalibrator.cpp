#include "stereoCalibrator.h"

using namespace std;
using namespace cv;

stereoCalibrator::stereoCalibrator(int image_pairs, int board_width, int board_height, float square_size)
{
    updateCaliConfig(image_pairs, board_width, board_height, square_size);
}

stereoCalibrator::~stereoCalibrator()
{
}

void stereoCalibrator::updateCaliConfig(int image_pairs, int board_width, int board_height, float square_size)
{
    // Update chessboard parameters.
    imagePairs = image_pairs;
    boardSize.height = board_height;
    boardSize.width = board_width;
    squareSize = square_size;

    // Calculate chessboard coordinates.
    for (int i = 0; i < boardSize.height; i++)
        for (int j = 0; j < boardSize.width; j++)
            objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));
}

void stereoCalibrator::calibrate(string caliImgPath1, string caliImgPath2, Size &imgSize, Mat &K1, Mat &D1, Mat &K2, Mat &D2, Mat &R, Mat &T, Mat &F, Mat &E)
{
    // Read images.
    vector<Mat> imgSet1, imgSet2;
    for (int i = 1; i <= imagePairs; i++)
    {
        string number = to_string(i);
        string suffix = ".bmp";
        imgSet1.push_back(imread(caliImgPath1 + number + suffix, IMREAD_GRAYSCALE));
        imgSet2.push_back(imread(caliImgPath2 + number + suffix, IMREAD_GRAYSCALE));
    }

    // Find corners.
    bool found1, found2;
    vector<vector<Point3f>> objectPointsSet;
    vector<vector<Point2f>> imgPoints1, imgPoints2;
    for (int i = 0; i < imagePairs; i++)
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
            cornerSubPix(imgSet1[i], cornerPoints1, Size(5, 5), Size(-1, -1), criteria);
            cornerSubPix(imgSet2[i], cornerPoints2, Size(5, 5), Size(-1, -1), criteria);

            objectPointsSet.push_back(objectPoints);

            imgPoints1.push_back(cornerPoints1);
            imgPoints2.push_back(cornerPoints2);
        }
    }
    stereoCalibrate(objectPointsSet, imgPoints1, imgPoints2, K1, D1, K2, D2, imgSet1[0].size(), R, T, E, F);
    imgSize = imgSet1[0].size();
}