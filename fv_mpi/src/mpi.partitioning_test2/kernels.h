#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include <vector>
#include "FVL/FVGlobal.h"
#include "FVL/FVMesh2D_SOA.h"
#include "FVL/FVArray.h"
using namespace FVL;

#define _choose(cond, true_val, false_val) ((cond) * (true_val) + (!(cond)) * (false_val))

enum FVEdgeType {
	FV_EDGE		= 0,
	FV_MPI_EDGE	= 1
};

void compute_flux(FVMesh2D_SOA &mesh, FVArray<double> &velocity, FVArray<double> &polution, FVArray<double> &flux, double dc);

void update(FVMesh2D_SOA &mesh, FVArray<double> &polution, FVArray<double> &flux, double dt);

#endif // _H_KERNELS_CPU
