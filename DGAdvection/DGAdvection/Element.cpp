

#ifndef MATH
#include <cmath>
#include<iostream>
#include<Eigen/Dense>
#include "Matrices.h"
#include "Element.h"
#endif 

using namespace Eigen;
extern const int N;

void Element::setNp(int i)
{
	Np = i;
}

int Element::getNp()
{
	return Np;
}

void Element::setRightIndex(int i)
{
	rightIndex = i;
}

int Element::getRightIndex()
{
	return rightIndex;
}

void Element::setLeftIndex(int i)
{
	leftIndex = i;
}

int Element::getLeftIndex()
{
	return leftIndex;
}

int Element::getLeftNormal()
{
	return leftNormal;
}
int Element::getRightNormal()
{
	return rightNormal;
}

int Element::getPrev()
{
	return *prev;
}
int Element::getNext()
{
	return *next;
}
void Element::setNext(int &i)
{
	next = &i;
}
void Element::setPrev(int &i)
{
	prev = &i;
}

ArrayXd Element::JacobiGL(int N) //returns GL nodes of order N (N+1 nodes for order N)
{
	ArrayXd returnArr = ArrayXd::Zero(N + 1);
	switch (N)
	{
	case 1:
		returnArr << -1,
			1;
		break;
	case 2:
		returnArr << -1,
			0,
			1;
		break;
	case 3:
		returnArr << -1,
			-0.447213595499957939282,
			0.447213595499957939282,
			1;
		break;
	case 4:
		returnArr << -1,
			-0.6546536707079771437983,
			0,
			0.654653670707977143798,
			1;
		break;
	case 5:
		returnArr << -1, -0.765055323929464692851, -0.2852315164806450963142, 0.2852315164806450963142, 0.765055323929464692851, 1;
		break;
	case 6:
		returnArr << -1, -0.830223896278566929872, -0.4688487934707142138038, 0, 0.468848793470714213804, 0.830223896278566929872, 1;
		break;
	case 7:
		returnArr << -1,
			-0.8717401485096066153375,
			-0.5917001814331423021445,
			-0.2092992179024788687687,
			0.2092992179024788687687,
			0.5917001814331423021445,
			0.8717401485096066153375,
			1;
		break;
	case 8:
		returnArr << -1,
			-0.8997579954114601573124,
			-0.6771862795107377534459,
			-0.3631174638261781587108,
			0,
			0.3631174638261781587108,
			0.6771862795107377534459,
			0.8997579954114601573124,
			1;
		break;
	case 9:
		returnArr << -1,
			-0.9195339081664588138289,
			-0.7387738651055050750031,
			-0.4779249498104444956612,
			-0.1652789576663870246262,
			0.1652789576663870246262,
			0.4779249498104444956612,
			0.7387738651055050750031,
			0.9195339081664588138289,
			1;

		break;
	case 10:
		returnArr << -1,
			-0.9340014304080591343323,
			-0.7844834736631444186224,
			-0.565235326996205006471,
			-0.2957581355869393914319,
			0,
			0.2957581355869393914319,
			0.565235326996205006471,
			0.7844834736631444186224,
			0.9340014304080591343323,
			1;
		break;
	}
	return returnArr;
}

ArrayXd Element::JacobiP(ArrayXd x, int alpha, int beta, int N)
{
	ArrayXd P0alphabeta;
	P0alphabeta = ArrayXd::Constant(x.size(), sqrt(pow(2.0, -alpha - beta - 1)*(tgamma(alpha + beta + 2) / (tgamma(alpha + 1)*tgamma(beta + 1)))));
	if (N == 0)
	{
		return P0alphabeta;
	}
	else if (N == 1)
	{
		ArrayXd P1alphabeta;
		P1alphabeta = (1.0 / 2.0)*P0alphabeta*sqrt((double)(alpha + beta + 3) / ((double)(alpha + 1)*(beta + 1)))*((alpha + beta + 2)*x + alpha - beta);
		return P1alphabeta;
	}
	else //use recurrence relation 
	{
		double aNM1, bNM1, aN;
		aNM1 = 2.0 / (2.0 * (N - 1) + alpha + beta) * sqrt((double)(N - 1)*(N - 1 + alpha + beta)*(N - 1 + alpha)*(N - 1 + beta) / ((2 * N + alpha + beta - 3)*(2 * N + alpha + beta - 1)));
		bNM1 = -(double)(alpha*alpha - beta * beta) / (double)((2 * (N - 1) + alpha + beta)*(2 * N + alpha + beta));
		aN = 2.0 / (2.0 * (N)+alpha + beta) * sqrt((double)N*(N + alpha + beta)*(N + alpha)*(N + beta) / ((2 * (N)+alpha + beta - 1)*(2 * (N)+alpha + beta + 1)));
		return ((x - bNM1) / aN) * JacobiP(x, alpha, beta, N - 1) - aNM1 / aN * JacobiP(x, alpha, beta, N - 2);
	}
}

ArrayXd Element::GradJacobiP(ArrayXd r, int alpha, int beta, int N)
{
	ArrayXd returnArr = ArrayXd::Zero(r.size());
	if (N != 0)
	{
		returnArr = sqrt(N*(N + alpha + beta + 1))*JacobiP(r, alpha + 1, beta + 1, N - 1);
	}
	return returnArr;
}

MatrixXd Element::Vandermonde(int N, ArrayXd r)
{
	MatrixXd V = MatrixXd::Zero(r.size(), N + 1);
	for (int i = 0; i < N + 1; i++)
	{
		V.col(i) = JacobiP(r, 0, 0, i);
	}
	return V;
}

MatrixXd Element::GradVandermonde(int N, ArrayXd r)
{
	MatrixXd DVr = MatrixXd::Zero(r.size(), N + 1);
	for (int i = 0; i < N + 1; i++)
	{
		DVr.col(i) = GradJacobiP(r, 0, 0, i);
	}
	return DVr;
}

MatrixXd Element::Dmatrix(int N,ArrayXd r, MatrixXd V)
{
	MatrixXd Vr = GradVandermonde(N,r);
	MatrixXd VT = V.transpose();
	MatrixXd VrT = Vr.transpose();
	MatrixXd DrT = VT.partialPivLu().solve(VrT);
	return DrT.transpose();
}

Element::Element()
{
	Np = N+1;
	leftNormal = -1;
	rightNormal = 1;
	index = 0;
	leftIndex = 0;
	rightIndex = 0;
	u = VectorXd::Zero(Np);
	xValues = ArrayXd::Zero(Np);

	//nodes = JacobiGL(N);
	//VanderMatrix = Vandermonde(N,nodes);
	//DiffMatrix = Dmatrix(N,nodes, VanderMatrix);
}

//Element::Element(int N, int indexNumber)
//{
//	Np = N+1;
//	leftNormal = -1;
//	rightNormal = 1;
//	index = indexNumber;
//	leftIndex = 0;
//	rightIndex = 0;
//	nodes = JacobiGL(N);
//	VanderMatrix = Vandermonde(N, nodes);
//	DiffMatrix = Dmatrix(N, nodes, VanderMatrix);
//}

void Element::print()
{
	std::cout
		<< "The index of this element is " << index << std::endl
		<< "The nodes of this index are" << '\n' << Element::nodes << std::endl
		<< "There are " << Element::getNp() << " nodes in this element" << std::endl
		<< "The left normal is " << Element::getLeftNormal() << std::endl
		<< "The right normal is " << Element::getRightNormal() << std::endl
		<< "The Vandermonde matrix is" << std::endl << Element::VanderMatrix << std::endl
		<< "The Differentiation matrix is" << std::endl << Element::DiffMatrix << std::endl
		<< "The xValues are" << std::endl << Element::xValues << std::endl
		<< "The u values are" << std::endl << Element::u << std::endl
		<< "The metric is " <<  Element::metric << std::endl
		<< "The left x value is " << Element::leftXValue << std::endl
		<< "The size of the element is " << Element::h << std::endl;

	if (next != nullptr)
	{
		std::cout << "the right element is element #" << Element::getNext() << std::endl;
	}
	if (prev != nullptr)
	{
		std::cout << "the left element is element #" << Element::getPrev() << std::endl;
	}
}
