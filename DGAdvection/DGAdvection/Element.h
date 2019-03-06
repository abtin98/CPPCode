#pragma once

#include <Eigen/Dense>
#include <iostream>
#include "Element.h"

using namespace Eigen;

class Element
{
private:
	int* next = nullptr;
	int* prev = nullptr;
	int leftNormal;
	int rightNormal;
	int leftIndex;
	int rightIndex;
	int Np;

	

public:
	double metric;
	double h;
	double leftXValue;
	double rightXValue;
	int index;
	MatrixXd DiffMatrix;
	MatrixXd VanderMatrix;
	ArrayXd nodes;
	ArrayXd xValues;
	VectorXd u;
		//int left;
		//int right;

		//int metric;
		//ArrayXd points;




		//ArrayXd linspace(double l, double r, int N);
	Element();
	//Element(int indexNumber);
	void setNp(int i);
	int getNp();
	void setRightIndex(int i);
	int getRightIndex();
	void setLeftIndex(int i);
	int getLeftIndex();
	int getLeftNormal();
	int getRightNormal();
	int getPrev();
	int getNext();
	void setNext(int &i);
	void setPrev(int &i);
	ArrayXd JacobiGL(int N);
	ArrayXd JacobiP(ArrayXd x, int alpha, int beta, int N);
	ArrayXd GradJacobiP(ArrayXd r, int alpha, int beta, int N);
	MatrixXd Vandermonde(int N, ArrayXd r);
	MatrixXd GradVandermonde(int N, ArrayXd r);
	MatrixXd Dmatrix(int N, ArrayXd r, MatrixXd V);
	void print();

	
};

