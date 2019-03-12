#include <iostream>
#include <Eigen/Dense>
#include "Element.h"
#include "Functions.h"

using namespace Eigen;

extern const int N;

ArrayXd linspace(double l, double r, int N)
{
	ArrayXd grid = ArrayXd::Zero(N);
	double dx = (r - l) / (double)(N - 1);
	for (int i = 0; i < N; i++)
	{
		grid(i) = l + dx * i;
	}
	return grid;
}

ArrayOfElements BuildGrid(double l, double r, int Nelements, int p, bool isPeriodic)
{
	ArrayOfElements v(Nelements);
	ArrayXd mesh = linspace(l, r, Nelements+1);
	v(0).nodes = v(0).JacobiGL(p);
	v(0).VanderMatrix = v(0).Vandermonde(p, v(0).nodes);
	v(0).DiffMatrix = v(0).Dmatrix(p, v(0).nodes, v(0).VanderMatrix);

	for (int i = 0; i < Nelements; i++)
	{

		v(i).leftXValue = mesh(i);
		v(i).h = mesh(i + 1) - mesh(i);
		v(i).metric = 2/v(i).h;
		v(i).index = i;
		if (i == 0)
		{
			v(i).xValues = v(i).leftXValue + ((1.0 + v(i).nodes) / 2.0) * v(i).h;
			continue;
		}
		v(i).nodes = v(0).nodes;
		v(i).VanderMatrix = v(0).VanderMatrix;
		v(i).DiffMatrix = v(0).DiffMatrix;
		v(i).xValues = v(i).leftXValue + ((1.0 + v(i).nodes) / 2.0) * v(i).h;
	}
	
	
	
	for (int i = 0; i < Nelements; i++)
	{
		if (i == 0 || i == Nelements-1)
		{
			continue;
		}
			
		v(i).setNext(v(i + 1).index);
		v(i).setPrev(v(i - 1).index);
	}
	v(0).setNext(v(1).index);
	v(Nelements - 1).setPrev(v(Nelements - 2).index);

	if (isPeriodic)
	{
		v(Nelements - 1).setNext(v(0).index);
		v(0).setPrev(v(Nelements - 1).index);
	}
	
	return v;
}

void printGrid(ArrayOfElements grid)
{
	for (int i = 0; i < grid.size(); i++)
	{
		grid(i).print();
		std::cout << "-----------------------" << std::endl;
	}
}

VectorXd getFlux(ArrayOfElements v, int index, double a, double alpha, double time, bool isPeriodic)
{
	VectorXd returnVec = VectorXd::Zero(v(0).getNp());
	if (isPeriodic)
	{
		returnVec(N) = a * v(index).u(N) - (0.5*(a*v(index).u(N) + a * v(v(index).getNext()).u(0))) - a * (1 - alpha) / 2.0 * (v(index).u(N)*v(index).getRightNormal() + v(v(index).getNext()).u(0)*v(v(index).getNext()).getLeftNormal());
		returnVec(0) = -a * v(index).u(0) + (0.5*(a*v(index).u(0) + a * v(v(index).getPrev()).u(N))) + a * (1 - alpha) / 2.0 * (v(v(index).getPrev()).u(N)*v(v(index).getPrev()).getRightNormal() + v(index).u(0)*v(index).getLeftNormal());
	}
	else if (!isPeriodic)
	{
		if (index != 0 || index != v.size() - 1)
		{
			returnVec(N) = a * v(index).u(N) - (0.5*(a*v(index).u(N) + a * v(v(index).getNext()).u(0))) + a * (1 - alpha) / 2.0 * (v(index).u(N)*v(index).getRightNormal() + v(v(index).getNext()).u(0)*v(v(index).getNext()).getLeftNormal());
			returnVec(0) = -a * v(index).u(0) + (0.5*(a*v(index).u(0) + a * v(v(index).getPrev()).u(N))) + a * (1 - alpha) / 2.0 * (v(v(index).getPrev()).u(N)*v(v(index).getPrev()).getRightNormal() + v(index).u(0)*v(index).getLeftNormal());
		}
		else if (index == 0)
		{
			double inflow = 5.0* exp(-5.0*pow(-time*a - 5, 2));
			returnVec(N) = a * v(index).u(N) - (0.5*(a*v(index).u(N) + a * v(v(index).getNext()).u(0))) + a * (1 - alpha) / 2.0 * (v(index).u(N)*v(index).getRightNormal() + v(v(index).getNext()).u(0)*v(v(index).getNext()).getLeftNormal());
			returnVec(0) = -a * v(index).u(0) + (0.5*(a*v(index).u(0) + a * inflow)) + a * (1 - alpha) / 2.0 * (inflow + v(index).u(0)*v(index).getLeftNormal());
		}
		else if (index == v.size() - 1)
		{
			returnVec(0) = -a * v(index).u(0) + (0.5*(a*v(index).u(0) + a * v(v(index).getPrev()).u(N))) + a * (1 - alpha) / 2.0 * (v(v(index).getPrev()).u(N)*v(v(index).getPrev()).getRightNormal() + v(index).u(0)*v(index).getLeftNormal());
		}
	}
	return  returnVec;
}

ArrayOfElements RHS(ArrayOfElements grid, double time, double a, double alpha, bool isPeriodic)
{
	VectorXd updatedU; //= VectorXd::Zero(grid(0).getNp());
	ArrayOfElements updatedGrid = grid;
	for (int i = 0; i < grid.size(); i++) //loops through each element
	{
		// calculate fluxes at boundaries
		VectorXd flux = getFlux(grid, i, a, alpha,time,isPeriodic);
		updatedU = grid(i).metric*grid(i).VanderMatrix*grid(i).VanderMatrix.transpose()*flux - a * grid(i).metric* grid(i).DiffMatrix *grid(i).u;
		updatedGrid(i).u = updatedU;
	}
	return updatedGrid;
}

MatrixXd ODESolvLSERK4(ArrayOfElements grid, double finalTime, double dt, double a, double alpha, bool isPeriodic)
{
	
	ArrayXd rk4a(5);
	ArrayXd rk4b(5);
	ArrayXd rk4c(5);

	rk4a << 0.0,
		-567301805773.0 / 1357537059087.0,
		-2404267990393.0 / 2016746695238.0,
		-3550918686646.0 / 2091501179385.0,
		-1275806237668.0 / 842570457699.0;

	rk4b << 1432997174477.0 / 9575080441755.0,
		5161836677717.0 / 13612068292357.0,
		1720146321549.0 / 2090206949498.0,
		3134564353537.0 / 4481467310338.0,
		2277821191437.0 / 14882151754819.0;

	rk4c << 0.0,
		1432997174477.0 / 9575080441755.0,
		2526269341429.0 / 6820363962896.0,
		2006345519317.0 / 3224310063776.0,
		2802321613138.0 / 2924317926251.0;



	double Nsteps = finalTime / dt;
	double time = 0;
	double timelocal;
	ArrayXd resu = ArrayXd::Zero(grid(0).getNp());
	std::cout << resu;
	ArrayOfElements rhs;
	//for (int tstep = 1; tstep <= Nsteps; tstep++)
	//{
	//	for (int INTRK = 0; INTRK < 5; INTRK++)
	//	{
	//		timelocal = time + (rk4c(INTRK)+1.0)*dt;
	//		rhs = RHS(grid,timelocal,a,alpha,isPeriodic);
	//		//grid = RHS;
	//		for (int i = 0; i < grid.size(); i++)
	//		{
	//			resu = rk4a(INTRK)*resu + dt * (rhs(i).u).array();
	//			grid(i).u = grid(i).u + rk4b(INTRK)*resu.matrix();
	//		}
	//	}
	//	//Increment time
	//	time = time + dt;
	//	//std::cout << time << std::endl;
	//}
	for (int tstep = 1; tstep <= Nsteps; tstep++)
	{
		rhs = RHS(grid, timelocal, a, alpha, isPeriodic);
		//printGrid(rhs);
		for (int i = 0; i < grid.size(); i++)
		{
			grid(i).u = grid(i).u +dt*rhs(i).u;
		}
		//printGrid(grid);
	}

	// Postprocess: returns only x and u values as final output.

	MatrixXd uOutput(grid(0).getNp(), grid.size());
	MatrixXd xOutput(grid(0).getNp(), grid.size());

	for (int i = 0; i < grid.size(); i++)
	{
		uOutput.col(i) = grid(i).u;
		xOutput.col(i) = grid(i).xValues;
	}
	Map<RowVectorXd> v1(uOutput.data(), uOutput.size());
	Map<RowVectorXd> v2(xOutput.data(), xOutput.size());
	MatrixXd output(2,v1.size());
	output.row(0) = v2;
	output.row(1) = v1;
	return output;
		
}