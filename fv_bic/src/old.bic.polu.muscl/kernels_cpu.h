#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "../ign.polu.common/kernels_common.h"

void cpu_bound_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVMat<double> &vecGradient, CFVArray<double> &cellPsi, double t, double dt);
void cpu_compute_unbounded_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &vs, CFVArray<double> &polution, CFVArray<double> &edgePsi, double dc);
void cpu_cellPsi(CFVMesh2D &mesh, CFVArray<double> &edgePsi, CFVArray<double> &cellPsi);
void cpu_bound_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &vs, CFVArray<double> &polution, double dc);
void cpu_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, double dt);

#endif // _H_KERNELS_CPU
