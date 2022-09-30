#include "phaseCalculator.h"

using namespace std;
using namespace cv;

phaseCalculator::phaseCalculator(const pmpStereoConfig &cfg)
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
        throw "Error image number!";

    relPhaseMap = Mat(stripImg[0].size(), CV_TYPE);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < stripImg[0].rows; ++i)
        for (int j = 0; j < stripImg[0].cols; ++j)
        {
            TYPE y, x, z;
            y = ROOT_THREE * (stripImg[0].at<uchar>(i, j) - stripImg[2].at<uchar>(i, j));
            x = 2.0 * stripImg[1].at<uchar>(i, j) - stripImg[0].at<uchar>(i, j) - stripImg[2].at<uchar>(i, j);
            z = atan2(y, x);
            relPhaseMap.at<TYPE>(i, j) = z;
        }
}

void phaseCalculator::calRelPhase_4step(const vector<Mat> &stripImg, Mat &relPhaseMap)
{
    // This pixel-to-pixel map may use SIMD or multithreading to improve proformance.

    // Check integrity of image set.
    if (stripImg.size() != 4)
        throw "Error image number!";

    relPhaseMap = Mat(stripImg[0].size(), CV_TYPE);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < stripImg[0].rows; ++i)
        for (int j = 0; j < stripImg[0].cols; ++j)
        {
            TYPE y, x, z;
            y = (TYPE)stripImg[3].at<uchar>(i, j) - (TYPE)stripImg[1].at<uchar>(i, j);
            x = (TYPE)stripImg[0].at<uchar>(i, j) - (TYPE)stripImg[2].at<uchar>(i, j);
            z = atan2(y, x);
            relPhaseMap.at<TYPE>(i, j) = z;
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
    return (phase1 < phase2) ? (z + PI_2) : z;
}

// Update pmp algorithm parameters.
void phaseCalculator::updateConfig(const pmpStereoConfig &cfg)
{
    // N-Step algorithm.
    shiftSteps = cfg.shiftSteps;
    heterodyneSteps = cfg.heterodyneSteps;
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

// Calculate absolute phase map.
void phaseCalculator::calAbsPhase(const vector<Mat> &relPhaseMap, Mat &absPhaseMap)
{
    // Check if relPhaseMap have correct phase maps.
    if (relPhaseMap.size() != 3)
        throw "The number of phase maps is wrong!";

    // Allocate memory for absPhaseMap.
    absPhaseMap.create(relPhaseMap[0].size(), CV_TYPE);

// Calculate heterodyne phase and use it to unwrap relative phase.
#pragma omp parallel for collapse(2)
    for (int i = 0; i < absPhaseMap.rows; ++i)
    {
        for (int j = 0; j < absPhaseMap.cols; ++j)
        {
            TYPE heterodynePhase;
            TYPE phase1 = relPhaseMap[0].at<TYPE>(i, j);
            TYPE phase2 = relPhaseMap[1].at<TYPE>(i, j);
            TYPE phase3 = relPhaseMap[2].at<TYPE>(i, j);
            heterodynePhase = (heterodyneSteps == TWO_STEP_HETERODYNE) ? calHeterodynePhase_2step(phase1, phase2, phase3) : calHeterodynePhase_3step(phase1, phase2, phase3);

            absPhaseMap.at<TYPE>(i, j) = phase1 + PI_2 * floor((heterodynePhase * ratio_3to1 - phase1) / PI_2 + FLOOR_PRECISION);
        }
    }
}