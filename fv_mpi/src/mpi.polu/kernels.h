#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include <vector>
#include "FVL/FVGlobal.h"
#include "FVL/FVMesh2D_SOA.h"
#include "FVL/FVMesh2D_SOA_Lite.h"
#include "FVL/FVArray.h"
using namespace FVL;

enum TAGS {
	TAG_LEFT_COMM,
	TAG_RIGHT_COMM,
	TAG_WRITER_SIZE,
	TAG_WRITER_INDEX,
	TAG_WRITER_POLU
};

#define _choose(cond, true_val, false_val) ((cond) * (true_val) + (!(cond)) * (false_val))

enum FVEdgeType {
	FV_EDGE		= 0,
	FV_MPI_EDGE	= 1
};

void communication(int id, int size, FVMesh2D_SOA_Lite &mesh, FVArray<double> &polution);

void compute_flux(FVMesh2D_SOA_Lite &mesh, FVArray<double> &flux, double dc);

void update(FVMesh2D_SOA_Lite &mesh, FVArray<double> &flux, double dt);

#endif // _H_KERNELS_CPU
