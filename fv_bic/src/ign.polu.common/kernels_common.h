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

void cpu_reverseA(CFVMesh2D &mesh, CFVMat<double> &matA);

/* compute system polution coeficients for system solve */
void cpu_compute_vecR(CFVMesh2D &mesh, CFVArray<double> &polution, CFVMat<double> &vecResult, double dc);

/* Compute vecABC */
void cpu_compute_gradient(CFVMesh2D &mesh, CFVMat<double> &matA, CFVMat<double> &vecResult, CFVMat<double> &vecABC);

double cpu_gradient_result(CFVMesh2D &mesh, CFVMat<double> &vecGradient, unsigned int edge, unsigned int cell, double t, double dt);

/* Compute initial u vectors */
void cpu_compute_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVMat<double> &vecGradient, double t, double dt);
void cpu_compute_border_u(CFVMesh2D &mesh, CFVRecons2D &recons, double dc);

#endif // _H_KERNELS_COMMON_