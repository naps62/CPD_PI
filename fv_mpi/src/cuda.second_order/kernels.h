#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/FVGlobal.h"
#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
using namespace FVL;

void compute_flux(
		CFVMesh2D &mesh,
		CFVArray<double> &velocity,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dc);

void update(
		CFVMesh2D &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dt);
#endif // _H_KERNELS_CPU
