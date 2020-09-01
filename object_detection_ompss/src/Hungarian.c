#include "Hungarian.h"
//#include <math.h>
double Solve(size_t trkNum, size_t detNum, float** DistMatrix, int* Assignment) {

	double cost = 0.0;

	double *distMatrixIn = (double *)malloc( sizeof(double) * trkNum * detNum);
	
	int *assignment = (int *)malloc(sizeof(size_t) * trkNum);
	
	size_t i;
	size_t j;
	
	for (i = 0; i < trkNum; i++)
		for (j = 0; j < detNum; j++)
			distMatrixIn[i + trkNum * j] = DistMatrix[i][j];
			
	assignmentoptimal(assignment, &cost, distMatrixIn, trkNum, detNum);		
	
	for (i = 0; i < trkNum; i++)
		Assignment[i] = assignment[i];
	
	free(distMatrixIn);
	free(assignment);
	

	return cost;
}


static void assignmentoptimal(int *assignment, double *cost, double *distMatrixIn, size_t nOfRows, size_t nOfColumns){
	
	double *distMatrix, *distMatrixTemp, *distMatrixEnd, *columnEnd, value, minValue;
	char *coveredColumns, *coveredRows, *starMatrix, *newStarMatrix, *primeMatrix;
	int nOfElements, minDim, row, col;

	/* initialization */
	*cost = 0;
	for (row = 0; row<nOfRows; row++)
		assignment[row] = -1;
		
	/* generate working copy of distance Matrix */
	/* check if all matrix elements are positive */
	nOfElements = nOfRows * nOfColumns;
	distMatrix = (double *)malloc(nOfElements * sizeof(double));
	distMatrixEnd = distMatrix + nOfElements;

	for (row = 0; row<nOfElements; row++)
	{
		value = distMatrixIn[row];
		if (value < 0)
			printf( "All matrix elements have to be non-negative.\n");
		distMatrix[row] = value;
	}
	
	/* memory allocation */
	coveredColumns = (char *)calloc(nOfColumns, sizeof(char));
	coveredRows = (char *)calloc(nOfRows, sizeof(char));
	starMatrix = (char *)calloc(nOfElements, sizeof(char));
	primeMatrix = (char *)calloc(nOfElements, sizeof(char));
	newStarMatrix = (char *)calloc(nOfElements, sizeof(char)); /* used in step4 */
	
	
	/* preliminary steps */
	if (nOfRows <= nOfColumns)
	{
		
		minDim = nOfRows;

		for (row = 0; row<nOfRows; row++)
		{
			/* find the smallest element in the row */
			distMatrixTemp = distMatrix + row;
			minValue = *distMatrixTemp;
			distMatrixTemp += nOfRows;
			while (distMatrixTemp < distMatrixEnd)
			{
				value = *distMatrixTemp;
				if (value < minValue)
					minValue = value;
				distMatrixTemp += nOfRows;
			}

			/* subtract the smallest element from each element of the row */
			distMatrixTemp = distMatrix + row;
			while (distMatrixTemp < distMatrixEnd)
			{
				*distMatrixTemp -= minValue;
				distMatrixTemp += nOfRows;
			}
		}

		/* Steps 1 and 2a */
		for (row = 0; row<nOfRows; row++)
			for (col = 0; col<nOfColumns; col++)
				if (fabs(distMatrix[row + nOfRows*col]) < __DBL_EPSILON__)
					if (!coveredColumns[col])
					{
						starMatrix[row + nOfRows*col] = 1;
						coveredColumns[col] = 1;
						break;
					}
	}
	else /* if(nOfRows > nOfColumns) */
	{
		minDim = nOfColumns;

		for (col = 0; col<nOfColumns; col++)
		{
			/* find the smallest element in the column */
			distMatrixTemp = distMatrix + nOfRows*col;
			columnEnd = distMatrixTemp + nOfRows;

			minValue = *distMatrixTemp++;
			while (distMatrixTemp < columnEnd)
			{
				value = *distMatrixTemp++;
				if (value < minValue)
					minValue = value;
			}

			/* subtract the smallest element from each element of the column */
			distMatrixTemp = distMatrix + nOfRows*col;
			while (distMatrixTemp < columnEnd)
				*distMatrixTemp++ -= minValue;
		}

		/* Steps 1 and 2a */
		for (col = 0; col<nOfColumns; col++)
			for (row = 0; row<nOfRows; row++)
				if (fabs(distMatrix[row + nOfRows*col]) < __DBL_EPSILON__)
					if (!coveredRows[row])
					{
						starMatrix[row + nOfRows*col] = 1;
						coveredColumns[col] = 1;
						coveredRows[row] = 1;
						break;
					}
		for (row = 0; row<nOfRows; row++)
			coveredRows[row] = 0;

	}
	
	/* move to step 2b */
	step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

	/* compute cost and remove invalid assignments */
	computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
		
	/* free allocated memory */
	free(distMatrix);
	free(coveredColumns);
	free(coveredRows);
	free(starMatrix);
	free(primeMatrix);
	free(newStarMatrix);

	return;
	
}


/********************************************************/
static void buildassignmentvector(int *assignment, char *starMatrix, int nOfRows, int nOfColumns)
{
	int row, col;

	for (row = 0; row<nOfRows; row++)
		for (col = 0; col<nOfColumns; col++)
			if (starMatrix[row + nOfRows*col])
			{
#ifdef ONE_INDEXING
				assignment[row] = col + 1; /* MATLAB-Indexing */
#else
				assignment[row] = col;
#endif
				break;
			}
}

/********************************************************/
void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows)
{
	int row, col;

	for (row = 0; row<nOfRows; row++)
	{
		col = assignment[row];
		if (col >= 0)
			*cost += distMatrix[row + nOfRows*col];
	}
}


/********************************************************/
static void step2a(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	char *starMatrixTemp, *columnEnd;
	int col;

	/* cover every column containing a starred zero */
	for (col = 0; col<nOfColumns; col++)
	{
		starMatrixTemp = starMatrix + nOfRows*col;
		columnEnd = starMatrixTemp + nOfRows;
		while (starMatrixTemp < columnEnd){
			if (*starMatrixTemp++)
			{
				coveredColumns[col] = 1;
				break;
			}
		}
	}

	/* move to step 3 */
	step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

}

/********************************************************/
static void step2b(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	int col, nOfCoveredColumns;

	/* count covered columns */
	nOfCoveredColumns = 0;
	for (col = 0; col<nOfColumns; col++)
		if (coveredColumns[col])
			nOfCoveredColumns++;

	if (nOfCoveredColumns == minDim)
	{
		/* algorithm finished */
		buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
	}
	else
	{
		/* move to step 3 */
		step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	}

}

/********************************************************/
static void step3(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	char zerosFound;
	int row, col, starCol;

	zerosFound = 1;
	while (zerosFound)
	{
		zerosFound = 0;
		for (col = 0; col<nOfColumns; col++)
			if (!coveredColumns[col])
				for (row = 0; row<nOfRows; row++)
					if ((!coveredRows[row]) && (fabs(distMatrix[row + nOfRows*col]) < __DBL_EPSILON__))
					{
						/* prime zero */
						primeMatrix[row + nOfRows*col] = 1;

						/* find starred zero in current row */
						for (starCol = 0; starCol<nOfColumns; starCol++)
							if (starMatrix[row + nOfRows*starCol])
								break;

						if (starCol == nOfColumns) /* no starred zero found */
						{
							/* move to step 4 */
							step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
							return;
						}
						else
						{
							coveredRows[row] = 1;
							coveredColumns[starCol] = 0;
							zerosFound = 1;
							break;
						}
					}
	}

	/* move to step 5 */
	step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	
}

/********************************************************/
static void step4(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
	int n, starRow, starCol, primeRow, primeCol;
	int nOfElements = nOfRows*nOfColumns;

	/* generate temporary copy of starMatrix */
	for (n = 0; n<nOfElements; n++)
		newStarMatrix[n] = starMatrix[n];

	/* star current zero */
	newStarMatrix[row + nOfRows*col] = 1;

	/* find starred zero in current column */
	starCol = col;
	for (starRow = 0; starRow<nOfRows; starRow++)
		if (starMatrix[starRow + nOfRows*starCol])
			break;

	while (starRow<nOfRows)
	{
		/* unstar the starred zero */
		newStarMatrix[starRow + nOfRows*starCol] = 0;

		/* find primed zero in current row */
		primeRow = starRow;
		for (primeCol = 0; primeCol<nOfColumns; primeCol++)
			if (primeMatrix[primeRow + nOfRows*primeCol])
				break;

		/* star the primed zero */
		newStarMatrix[primeRow + nOfRows*primeCol] = 1;

		/* find starred zero in current column */
		starCol = primeCol;
		for (starRow = 0; starRow<nOfRows; starRow++)
			if (starMatrix[starRow + nOfRows*starCol])
				break;
	}

	/* use temporary copy as new starMatrix */
	/* delete all primes, uncover all rows */
	for (n = 0; n<nOfElements; n++)
	{
		primeMatrix[n] = 0;
		starMatrix[n] = newStarMatrix[n];
	}
	for (n = 0; n<nOfRows; n++)
		coveredRows[n] = 0;

	/* move to step 2a */
	step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
static void step5(int *assignment, double *distMatrix, char *starMatrix, char *newStarMatrix, char *primeMatrix, char *coveredColumns, char *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	double h, value;
	int row, col;

	/* find smallest uncovered element h */
	h = __DBL_MAX__;
	for (row = 0; row<nOfRows; row++)
		if (!coveredRows[row])
			for (col = 0; col<nOfColumns; col++)
				if (!coveredColumns[col])
				{
					value = distMatrix[row + nOfRows*col];
					if (value < h)
						h = value;
				}

	/* add h to each covered row */
	for (row = 0; row<nOfRows; row++)
		if (coveredRows[row])
			for (col = 0; col<nOfColumns; col++)
				distMatrix[row + nOfRows*col] += h;

	/* subtract h from each uncovered column */
	for (col = 0; col<nOfColumns; col++)
		if (!coveredColumns[col])
			for (row = 0; row<nOfRows; row++)
				distMatrix[row + nOfRows*col] -= h;

	/* move to step 3 */
	step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}


