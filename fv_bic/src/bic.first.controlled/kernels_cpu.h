#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/FVGlobal.h"
#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
using namespace FVL;

void cpu_reverseA(
		CFVMesh2D &mesh,
		CFVMat<double> &matA);

/* compute system polution coeficients for system solve */
void cpu_vecResult(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVMat<double> &vecResult,
		double dc);

/* Compute vecABC */
void cpu_vecABC(
		CFVMesh2D &mesh,
		CFVMat<double> &matA,
		CFVMat<double> &vecResult,
		CFVMat<double> &vecABC);

void cpu_compute_unbounded_flux(
		CFVMesh2D &mesh,
		CFVArray<double> &velocity,
		CFVMat<double> &vecABC,
		CFVArray<double> &polution,
		CFVArray<double> &partial_flux,
		CFVArray<double> &edgePsi,
		double dc);

void cpu_cellPsi(
		CFVMesh2D &mesh,
		CFVArray<double> &edgePsi,
		CFVArray<double> &cellPsi);

void cpu_bound_flux(
		CFVMesh2D &mesh,
		CFVArray<double> &velocity,
		CFVArray<double> &cellPsi,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dc);

void cpu_update(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dt);
#endif // _H_KERNELS_CPU
