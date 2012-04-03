#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/FVGlobal.h"
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
		double dc);

/* Compute vecABC */
void cpu_compute_vecABC(
		CFVMesh2D &mesh,
		CFVMat<double> &matA,
		CFVMat<double> &vecResult,
		CFVMat<double> &vecABC);

/**
 * For each edge, compute system result for each edge.
 * If system_result of a cell is not between average value of both neighbor cells, negates the system for that cell, so that average value is used instead
 */
void cpu_validate_ABC(
		CFVMesh2D &mesh,
		CFVMat<double> &vecABC,
		CFVArray<double> &polution,
		CFVArray<int> &vecValidABC,
		double dc);

void cpu_compute_flux(
		CFVMesh2D &mesh,
		CFVArray<double> &velocity,
		CFVMat<double> &vecABC,
		CFVArray<int> &vecValidABC,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dc);

void cpu_update(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dt);
#endif // _H_KERNELS_CPU
