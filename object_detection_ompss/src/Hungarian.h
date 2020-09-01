#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <float.h>

//#define DBL_EPSILON 2.2204460492503131E-16
//#define __DBL_EPSILON__ DBL_EPSILON

double Solve(size_t trkNum, size_t detNum,float** DistMatrix, int* Assignment);

static void assignmentoptimal(int *assignment, double *cost, double *distMatrixIn, size_t nOfRows, size_t nOfColumns);
static void buildassignmentvector(int *assignment, char *starMatrix, int nOfRows, int nOfColumns);
static void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows);
static void step2a(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim);
static void step2b(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim);
static void step3(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim);
static void step4(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
static void step5(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim);

#endif //HUNGARIAN_H
