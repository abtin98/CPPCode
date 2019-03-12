#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "Matrices.h"
#include "Element.h"
#include "Functions.h"
#include <fstream>
#include <cstdlib>

extern const int N;
using namespace Eigen;

int main()
{
	int NumberOfElements = 100;
	double a = 0.5;
	double alpha = 0.0;
	double finalTime = 2.0;
	double dt = 0.005;
	bool isPeriodic = true;
	ArrayOfElements grid = BuildGrid(0.0, 10.0,NumberOfElements, N, isPeriodic);
	for (int i = 0; i < NumberOfElements; i++) //initialize grid with initial conditions
	{
		grid(i).u = 0.5*exp(-1.0*pow(grid(i).xValues - 5.0, 2));
	}
	
	//grid(0).u(0) = 0.0;
	//grid(0).u(1) = 1.33;
	//grid(1).u(0) = 2;
	//grid(1).u(1) = 5;

	//printGrid(grid);

	//grid = RHS(grid, 0.0, a, alpha, isPeriodic);

	//printGrid(grid);
	//std::cout << "Element 0" << std::endl << getFlux(grid, 0, a, alpha, 0.0, isPeriodic) << std::endl
	//	      << "Element 1" << std::endl << getFlux(grid, 1, a, alpha, 0.0, isPeriodic) << std::endl;

	MatrixXd out = ODESolvLSERK4(grid, finalTime, dt, a, alpha, isPeriodic);

	

	std::ofstream myfile("data.txt", std::ios::trunc);

	for (int i = 0; i < out.cols(); i++)
	{
		for (int j = 0; j < out.rows(); j++)
		{
			myfile << out(j, i) << " ";
		}
		myfile << std::endl;
	}
	

	myfile.close();
	


	

	return 0;
}