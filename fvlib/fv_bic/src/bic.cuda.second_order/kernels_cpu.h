#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
using namespace FVL;

void cpu_compute_reverseA(
		CFVMesh2D &mesh,
		CFVMat<double> &matA);

/* compute system polution coeficients for system solve */
void cpu_compute_vecResult(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVMat<double> &vecResult,
		CFVArray<double> velocity,
		double dc);

/* Compute vecABC */
void cpu_compute_vecABC(
		CFVMesh2D &mesh,
		CFVMat<double> &matA,
		CFVMat<double> &vecResult,
		CFVMat<double> &vecABC);

void cpu_compute_flux(
		CFVMesh2D &mesh,
		CFVArray<double> &velocity,
		CFVMat<double> &vecABC,
		CFVArray<double> &flux,
		double dc);

void cpu_update(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dt);
#endif // _H_KERNELS_CPU
