#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "Matrices.h"
#include "Element.h"
#include "Functions.h"

extern const int N;
using namespace Eigen;

int main()
{
	int NumberOfElements = 4;
	bool isPeriodic = true;
	ArrayOfElements grid = BuildGrid(0.0, 10.0,NumberOfElements, N, isPeriodic);
	for (int i = 0; i < NumberOfElements; i++) //initialize grid with initial conditions
	{
		grid(i).u = 5.0*exp(-5.0*pow(grid(i).xValues - 5.0, 2));
	}

	printGrid(grid);

	grid = RHS(grid, 0, 0.5, 0.5, isPeriodic);

	printGrid(grid);

	return 0;
}