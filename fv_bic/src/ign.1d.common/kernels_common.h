#ifndef _H_KERNELS_COMMON_
#define _H_KERNELS_COMMON_

#include "FVL/FVGlobal.h"
#include "FVL/CFVMesh2D.h"
#include "FVL/CFVRecons2D.h"
#include "FVL/CFVArray.h"

using namespace FVL;

// TODO: convert to cuda
double cpu_compute_mesh_parameter(CFVMesh2D &mesh);

void cpu_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max);

double _min(double x, double y);
double _max(double x, double y);


void cpu_compute_a(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &vecA);

#endif // _H_KERNELS_COMMON_