//#pragma once
#include<Eigen/Dense>
#include "Element.h"
using namespace Eigen;

ArrayXd linspace(double l, double r, int N);
ArrayXd JacobiGL(int N);
ArrayXd JacobiP(ArrayXd x, int alpha, int beta, int N);
ArrayXd GradJacobiP(ArrayXd r, int alpha, int beta, int N);
MatrixXd Vandermonde(int N, ArrayXd r);
MatrixXd GradVandermonde(int N, ArrayXd r);
MatrixXd Dmatrix(int N, ArrayXd r, MatrixXd V);