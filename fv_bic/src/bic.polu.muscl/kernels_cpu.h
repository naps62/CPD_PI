#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/FVGlobal.h"
#include "FVL/CFVMesh2D.h"
#include "FVL/CFVRecons2D.h"
#include "FVL/CFVArray.h"

using namespace FVL;


// TODO: convert to cuda
double cpu_compute_mesh_parameter(CFVMesh2D &mesh);

void cpu_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max);

void cpu_reverseA(CFVMesh2D &mesh, CFVMat<double> &matA);

/* compute system polution coeficients for system solve */
void cpu_compute_vecR(CFVMesh2D &mesh, CFVArray<double> &polution, CFVMat<double> &vecResult, double dc);

/* Compute vecABC */
void cpu_compute_gradient(CFVMesh2D &mesh, CFVMat<double> &matA, CFVMat<double> &vecResult, CFVMat<double> &vecABC);

/* Compute initial u vectors */
void cpu_compute_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVMat<double> &vecGradient, double t, double dt);
void cpu_compute_border_u(CFVMesh2D &mesh, CFVRecons2D &recons, double dc);

void cpu_bound_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVMat<double> &vecGradient, CFVArray<double> &cellPsi, double t, double dt);
void cpu_compute_unbounded_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &vs, CFVArray<double> &polution, CFVArray<double> &edgePsi, double dc);
void cpu_cellPsi(CFVMesh2D &mesh, CFVArray<double> &edgePsi, CFVArray<double> &cellPsi);
void cpu_bound_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &vs, CFVArray<double> &polution, double dc);
void cpu_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, double dt);











void cpu_compute_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &velocity);
//bool cpu_bad_cell_detector(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution);
void cpu_fix_u(CFVMesh2D &mesh,CFVRecons2D &recons, CFVArray<double> &polution);
void cpu_fix_border_u(CFVMesh2D &mesh, CFVRecons2D &recons, double dc);
void cpu_fix_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &velocity);
void cpu_fix_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, double dt);

#endif // _H_KERNELS_CPU
