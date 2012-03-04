#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/CFVMesh2D.h"
#include "FVL/CFVVect.h"
using namespace FVL;

void cpu_compute_reverseA(
		CFVMesh2D &mesh,
		CFVMat<double> &matA);

/* compute system polution coeficients for system solve */
void cpu_compute_vecResult(
		CFVMesh2D &mesh,
		CFVVect<double> &polution,
		CFVMat<double> &vecResult);

/* Compute vecABC */
void cpu_compute_vecABC(
		CFVMesh2D &mesh,
		CFVMat<double> &matA,
		CFVMat<double> &vecResult,
		CFVMat<double> &vecABC);

void cpu_compute_flux(
		CFVMesh2D &mesh,
		CFVVect<double> &polution,
		CFVVect<double> &velocity,
		CFVMat<double> &vecABC,
		CFVMat<double> &flux,
		double dc);

void cpu_update(
		CFVMesh2D &mesh,
		CFVVect<double> &polution,
		CFVMat<double> &flux,
		double dt);
#endif // _H_KERNELS_CPU
