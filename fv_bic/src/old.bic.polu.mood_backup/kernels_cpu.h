#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/FVGlobal.h"
#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"

using namespace FVL;


// TODO: convert to cuda
double cpu_compute_mesh_parameter(CFVMesh2D &mesh);

void cpu_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max);

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

void cpu_compute_flux(
		CFVMesh2D &mesh,
		CFVArray<double> &velocity,
		CFVMat<double> &vecABC,
		CFVArray<double> &polution,
		CFVArray<double> &partial_flux,
		double dc, double t,double dt);

void cpu_update(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dt);

void cpu_reset_oldflux(CFVArray<double> &oldflux);

void cpu_detect_polution_errors(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		CFVArray<double> &oldflux,
		CFVArray<bool> &invalidate_flux);

void cpu_fix_polution_errors(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &velocity,
		CFVArray<double> &flux,
		CFVArray<double> &oldflux,
		CFVArray<bool> &invalidate_flux);

void cpu_fix_update(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		CFVArray<double> &oldflux,
		double dt,
		CFVArray<bool> &invalidate_flux);

#endif // _H_KERNELS_CPU
