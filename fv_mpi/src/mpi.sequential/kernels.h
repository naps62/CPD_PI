#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/FVGlobal.h"
#include "FVL/FVMesh2D_SOA.h"
#include "FVL/CFVArray.h"
using namespace FVL;

void compute_flux(
		FVMesh2D_SOA &mesh,
		CFVArray<double> &velocity,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dc);

void update(
		FVMesh2D_SOA &mesh,
		CFVArray<double> &polution,
		CFVArray<double> &flux,
		double dt);
#endif // _H_KERNELS_CPU
