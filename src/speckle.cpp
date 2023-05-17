#include "speckle.h"

using namespace std;

speckle::speckle(int _winSize, int _maxDisparity = 64)
{
    if (winSize < 3)
    {
        cout << "Window size can not less than 3!" << endl;
        throw exception();
    }
    maxDisparity = _maxDisparity;
    winSize = _winSize;
    winArea = _winSize * _winSize;
}

speckle::~speckle()
{
}

void speckle::DB(cv::Mat &src, cv::Mat &dst)
{
    // Create the empty binary array.
    dst.create(cv::Size(src.cols, src.rows), CV_8U);
    int halfSize = winSize / 2;
    cv::Rect imgBoundary(0, 0, src.rows, src.cols);
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            unsigned int sum = 0;
            for (int dy = -halfSize; dy <= halfSize; dy++)
            {
                for (int dx = -halfSize; dx <= halfSize; dx++)
                {
                    int u = i + dy;
                    int v = j + dx;
                    bool inside = imgBoundary.contains(cv::Point(u, v));
                    sum += inside ? src.at<uchar>(u, v) : 0;
                }
            }
            dst.at<uchar>(i, j) = (winArea * src.at<uchar>(i, j) > sum) ? 1 : 0;
        }
    }
}

void speckle::match(cv::Mat &src1, cv::Mat &src2, cv::Mat &disparity)
{
    // Create the empty cost array.
    disparity.create(cv::Size(src1.cols, src1.rows), CV_8U);
    int halfSize = winSize / 2;
    cv::Rect imgBoundary(0, 0, src1.rows, src1.cols);
    for (int i = 0; i < src1.rows; i++)
    {
        for (int j = 0; j < src1.cols; j++)
        {
            u_int16_t minHamingDis = winArea;
            u_int8_t dis = 0;
            for (int k = 0; k < maxDisparity; k++)
            {
                u_int16_t hamingDis = 0;
                for (int dy = -halfSize; dy <= halfSize; dy++)
                {
                    for (int dx = -halfSize; dx <= halfSize; dx++)
                    {
                        int u = i + dy;
                        int v1 = j + dx;
                        int v2 = j + dx + k;
                        bool inside1 = imgBoundary.contains(cv::Point(u, v1));
                        bool inside2 = imgBoundary.contains(cv::Point(u, v2));

                        if (inside1 & inside2)
                            hamingDis += src1.at<u_int8_t>(u, v1) == src2.at<u_int8_t>(u, v2);
                        else if (inside1 ^ inside2)
                            hamingDis += inside1? (src1.at<u_int8_t>(u, v1) == 0) : (src2.at<u_int8_t>(u, v2) == 0);
                    }
                }
                if (hamingDis < minHamingDis)
                {
                    minHamingDis = hamingDis;
                    dis = k;
                }
            }
            disparity.at<u_int8_t>(i, j) = dis;
        }
    }
}