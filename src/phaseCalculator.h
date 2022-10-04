#ifndef PHASE_CALCULATOR
#define PHASE_CALCULATOR

#include <iostream>
#include <vector>
#include "setting.h"

// Struct to initialize phase calculator.
struct pmpConfig
{
    bool shiftSteps; // N-Step algorithm.
    // Steps to calculate final hetero phase map.
    // 0 for 2-step method. phase1 phase2 -> phase12, phase12 phase3 -> phase123.
    // 1 for 3-step method. phase1 phase2 -> phase12, phase2 phase3 -> phase23, phase12 phase23 -> phase123.
    bool heterodyneSteps;
    // Degree of modulation threshold.
    TYPE BTH;
    TYPE freq1, freq2, freq3; // Strip frequency.

    pmpConfig(TYPE freq1, TYPE freq2, TYPE freq3, TYPE B_th, bool shiftSteps, bool heterodyneSteps);
};

// Calculator for phase calculation.
class phaseCalculator
{
protected:
    // Wave-length ratio.
    TYPE ratio_3to2, ratio_2to1, ratio_3to1;
    // N-Step algorithm.
    bool shiftSteps;
    // Steps to calculate final heterodyne phase map.
    // 0 for 2-step method. phase1 phase2 -> phase12, phase12 phase3 -> phase123.
    // 1 for 3-step method. phase1 phase2 -> phase12, phase2 phase3 -> phase23, phase12 phase23 -> phase123.
    bool heterodyneSteps;
    // Degree of modulation threshold.
    TYPE BTH;

    // Calculate relative phase according to phase shift steps.
    void calRelPhase_3step(const std::vector<cv::Mat> &stripImg, cv::Mat &relPhaseMap);
    void calRelPhase_4step(const std::vector<cv::Mat> &stripImg, cv::Mat &relPhaseMap);

    // Calcuate heterodyne phase.
    TYPE calHeterodynePhase_2step(const TYPE &phase1, const TYPE &phase2, const TYPE &phase3) const;
    TYPE calHeterodynePhase_3step(const TYPE &phase1, const TYPE &phase2, const TYPE &phase3) const;

    // Calculate heterodyne phase.
    TYPE heterodyne(const TYPE &phase1, const TYPE &phase2) const;

public:
    // Constructor.
    phaseCalculator(const pmpConfig &cfg);
    // Destructor.
    ~phaseCalculator();

    // Update pmp algorithm parameters.
    void updateConfig(const pmpConfig &cfg);

    // Calculate relative phase map.
    void calRelPhase(const std::vector<cv::Mat> &stripImg, cv::Mat &relPhaseMap);
    // Calculate heterodyne phase map.
    void calHeterodynePhase(const std::vector<cv::Mat> &relPhaseMap, cv::Mat &hetetodynePhaseMap);

    // Calculate absolute phase map.
    void calAbsPhase(const std::vector<cv::Mat> &relPhaseMap, cv::Mat &absPhaseMap, bool filter = true);
};

#endif