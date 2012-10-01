#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "../ign.1d.common/kernels_common.h"

void cpu_compute_p(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &p);
void cpu_compute_u(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, CFVArray<double> &vecA, double CFL);

void cpu_compute_flux(CFVMesh2D &mesh, CFVArray<double> &vs, CFVRecons2D &recons);
void cpu_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, double dt);

#endif // _H_KERNELS_CPU
