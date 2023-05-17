#ifndef SPECKLE
#define SPECKLE

#include "setting.h"

class speckle
{
protected:
    int winSize;
    int winArea;
    int maxDisparity;
public:
    speckle(int _winSize, int _maxDisparity = 64);
    ~speckle();
    
    // Implementatation of DB algorithm.
    void DB(cv::Mat &src, cv::Mat &dst);
    void match(cv::Mat &src1, cv::Mat &src2, cv::Mat &disparity);
};

#endif