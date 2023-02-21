#include "phaseCalculator.h"

using namespace std;
using namespace cv;

pmpConfig::pmpConfig(TYPE freq1, TYPE freq2, TYPE freq3, TYPE B_th, bool shiftSteps, bool heterodyneSteps) : freq1(freq1),
                                                                                                             freq2(freq2),
                                                                                                             freq3(freq3),
                                                                                                             BTH(B_th),
                                                                                                             shiftSteps(shiftSteps),
                                                                                                             heterodyneSteps(heterodyneSteps)
{
}

phaseCalculator::phaseCalculator(const pmpConfig &cfg)
{
    updateConfig(cfg);
}

phaseCalculator::~phaseCalculator()
{
}

// Calcuate relative phase according to phase shift steps.
void phaseCalculator::calRelPhase_3step(const vector<Mat> &stripImg, Mat &relPhaseMap)
{
    // This pixel-to-pixel map may use SIMD or multithreading to improve proformance.

    // Check integrity of image set.
    if (stripImg.size() != 3)
    {
        cout << "Error image number!" << endl;
        throw exception();
    }

    relPhaseMap = Mat(stripImg[0].size(), CV_TYPE);
#pragma omp parallel for simd
    for (int i = 0; i < stripImg[0].rows; ++i)
        for (int j = 0; j < stripImg[0].cols; ++j)
        {
            uchar I1, I2, I3;
            TYPE y, x, b;
            I1 = stripImg[0].at<uchar>(i, j);
            I2 = stripImg[1].at<uchar>(i, j);
            I3 = stripImg[2].at<uchar>(i, j);
            y = ROOT_THREE * (I1 - I2);
            x = 2.0 * I2 - I1 - I3;
            b = TWO_THIRD * sqrt(pow(y, 2) + pow(x, 2));
            if (b < BTH)
                relPhaseMap.at<TYPE>(i, j) = NAN;
            else
                relPhaseMap.at<TYPE>(i, j) = atan2(y, x);
        }
}

void phaseCalculator::calRelPhase_4step(const vector<Mat> &stripImg, Mat &relPhaseMap)
{
    // This pixel-to-pixel map may use SIMD or multithreading to improve proformance.

    // Check integrity of image set.
    if (stripImg.size() != 4)
    {
        cout << "Error image number!" << endl;
        throw exception();
    }

    relPhaseMap = Mat(stripImg[0].size(), CV_TYPE);

#pragma omp parallel for simd
    for (int i = 0; i < stripImg[0].rows; ++i)
        for (int j = 0; j < stripImg[0].cols; ++j)
        {
            uchar I1, I2, I3, I4;
            TYPE y, x, b;
            I1 = stripImg[0].at<uchar>(i, j);
            I2 = stripImg[1].at<uchar>(i, j);
            I3 = stripImg[2].at<uchar>(i, j);
            I4 = stripImg[3].at<uchar>(i, j);
            y = I4 - I2;
            x = I1 - I3;
            b = 0.5 * sqrt(pow(y, 2) + pow(x, 2));
            if (b < BTH)
                relPhaseMap.at<TYPE>(i, j) = NAN;
            else
                relPhaseMap.at<TYPE>(i, j) = atan2(y, x);
        }
}

// Calcuate heterodyne phase according to phase shift steps.
TYPE phaseCalculator::calHeterodynePhase_2step(const TYPE &phase1, const TYPE &phase2, const TYPE &phase3) const
{
    // This pixel-to-pixel map may use SIMD or multithreading to improve proformance.
    TYPE phase12, phase123;
    phase12 = heterodyne(phase1, phase2);
    phase123 = heterodyne(phase12, phase3);
    return phase123;
}

TYPE phaseCalculator::calHeterodynePhase_3step(const TYPE &phase1, const TYPE &phase2, const TYPE &phase3) const
{
    // This pixel-to-pixel map may use SIMD or multithreading to improve proformance.
    TYPE phase13, phase23, phase123;
    phase13 = heterodyne(phase1, phase3);
    phase23 = heterodyne(phase2, phase3);
    phase123 = heterodyne(phase13, phase23);
    return phase123;
}

TYPE phaseCalculator::heterodyne(const TYPE &phase1, const TYPE &phase2) const
{
    TYPE z = phase1 - phase2;
    return (z < 0) ? (z + PI_2) : z;
}

// Update pmp algorithm parameters.
void phaseCalculator::updateConfig(const pmpConfig &cfg)
{
    // N-Step algorithm.
    shiftSteps = cfg.shiftSteps;
    heterodyneSteps = cfg.heterodyneSteps;
    BTH = cfg.BTH;
    // Calcuate wave length ratios.YPE
    if (heterodyneSteps == TWO_STEP_HETERODYNE)
    {
        ratio_2to1 = cfg.freq1 / (cfg.freq1 - cfg.freq2);
        ratio_3to2 = cfg.freq1 - cfg.freq2;
        ratio_3to1 = cfg.freq1;
    }
    else
    {
        ratio_2to1 = cfg.freq1 / (cfg.freq1 - cfg.freq3);
        ratio_3to2 = cfg.freq1 - cfg.freq3;
        ratio_3to1 = cfg.freq1;
    }
}

// Calculate relative phase map.
void phaseCalculator::calRelPhase(const vector<Mat> &stripImg, Mat &relPhaseMap)
{
    if (shiftSteps == FOUR_STEP_SHIFT)
        calRelPhase_4step(stripImg, relPhaseMap);
    else
        calRelPhase_3step(stripImg, relPhaseMap);
}

void phaseCalculator::calHeterodynePhase(const std::vector<cv::Mat> &relPhaseMap, cv::Mat &hetetodynePhaseMap)
{
    // Allocate memory for hetetodynePhaseMap.
    hetetodynePhaseMap.create(relPhaseMap[0].size(), CV_TYPE);

// Calculate heterodyne phase and use it to unwrap relative phase.
#pragma omp parallel for simd
    for (int i = 0; i < hetetodynePhaseMap.rows; ++i)
    {
        for (int j = 0; j < hetetodynePhaseMap.cols; ++j)
        {
            TYPE phase123;
            TYPE phase1 = relPhaseMap[0].at<TYPE>(i, j);
            TYPE phase2 = relPhaseMap[1].at<TYPE>(i, j);
            TYPE phase3 = relPhaseMap[2].at<TYPE>(i, j);

            if (isnan(phase1) | isnan(phase2) | isnan(phase3))
                hetetodynePhaseMap.at<TYPE>(i, j) = NAN;
            else
            {
                if (heterodyneSteps == TWO_STEP_HETERODYNE)
                {
                    TYPE phase12, phase123, absPhase12;
                    phase12 = heterodyne(phase1, phase2);
                    phase123 = heterodyne(phase12, phase3);
                    absPhase12 = phase12 + PI_2 * round((phase123 * ratio_3to2 - phase12) / PI_2);
                    hetetodynePhaseMap.at<TYPE>(i, j) = absPhase12;
                }
                else
                {
                    TYPE absPhase13, phase13, phase23, phase123;
                    phase13 = heterodyne(phase1, phase3);
                    phase23 = heterodyne(phase2, phase3);
                    phase123 = heterodyne(phase13, phase23);
                    absPhase13 = phase13 + PI_2 * round((phase123 * ratio_3to2 - phase13) / PI_2);
                    hetetodynePhaseMap.at<TYPE>(i, j) = absPhase13;
                }
            }
        }
    }
}

// Calculate absolute phase map.
void phaseCalculator::calAbsPhase(const vector<Mat> &relPhaseMap, Mat &absPhaseMap, bool filter)
{
    // Check if relPhaseMap have correct phase maps.
    if (relPhaseMap.size() != 3)
    {
        cout << "The number of phase maps is wrong!" << endl;
        throw exception();
    }

    // Allocate memory for absPhaseMap.
    absPhaseMap.create(relPhaseMap[0].size(), CV_TYPE);

// Calculate heterodyne phase and use it to unwrap relative phase.
#pragma omp parallel for simd
    for (int i = 0; i < absPhaseMap.rows; ++i)
    {
        for (int j = 0; j < absPhaseMap.cols; ++j)
        {
            TYPE phase123;
            TYPE phase1 = relPhaseMap[0].at<TYPE>(i, j);
            TYPE phase2 = relPhaseMap[1].at<TYPE>(i, j);
            TYPE phase3 = relPhaseMap[2].at<TYPE>(i, j);

            if (isnan(phase1) | isnan(phase2) | isnan(phase3))
                absPhaseMap.at<TYPE>(i, j) = NAN;
            else
            {
                if (heterodyneSteps == TWO_STEP_HETERODYNE)
                {
                    TYPE phase12, phase123, absPhase12;
                    phase12 = heterodyne(phase1, phase2);
                    phase123 = heterodyne(phase12, phase3);
                    absPhase12 = phase12 + PI_2 * round((phase123 * ratio_3to2 - phase12) / PI_2);
                    absPhaseMap.at<TYPE>(i, j) = phase1 + PI_2 * round((absPhase12 * ratio_2to1 - phase1) / PI_2);
                }
                else
                {
                    TYPE absPhase13, phase13, phase23, phase123;
                    phase13 = heterodyne(phase1, phase3);
                    phase23 = heterodyne(phase2, phase3);
                    phase123 = heterodyne(phase13, phase23);
                    absPhase13 = phase13 + PI_2 * round((phase123 * ratio_3to2 - phase13) / PI_2);
                    absPhaseMap.at<TYPE>(i, j) = phase1 + PI_2 * round((absPhase13 * ratio_2to1 - phase1) / PI_2);
                }
            }
        }
    }
    if (filter)
        medianBlur(absPhaseMap, absPhaseMap, PHASE_FILTER_WINSIZE);
}