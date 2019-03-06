#pragma once
#include <iostream>
#include <Eigen/Dense>
#include "Element.h"

using namespace Eigen;

typedef Matrix<Element, Dynamic, 1> ArrayOfElements;
typedef Matrix<Element, Dynamic, Dynamic> MatrixOfElements;

ArrayXd linspace(double l, double r, int num);
ArrayOfElements BuildGrid(double l, double r, int num, int Np, bool isPeriodic);
void printGrid(ArrayOfElements grid);
VectorXd getFlux(ArrayOfElements v, int index, double a, double alpha, double time, bool isPeriodic);
ArrayOfElements RHS(ArrayOfElements grid, double time, double a, double alpha, bool isPeriodic);
MatrixXd ODESolvLSERK4(ArrayOfElements grid, double finalTime, double dt, double a, double alpha, bool isPeriodic);