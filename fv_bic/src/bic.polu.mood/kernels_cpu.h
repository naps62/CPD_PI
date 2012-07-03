#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "../ign.polu.common/kernels_common.h"

void cpu_compute_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &velocity);
void cpu_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, double dt);
bool cpu_bad_cell_detector(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution);
void cpu_fix_u(CFVMesh2D &mesh,CFVRecons2D &recons, CFVArray<double> &polution);
void cpu_fix_border_u(CFVMesh2D &mesh, CFVRecons2D &recons, double dc);
void cpu_fix_flux(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &velocity);
void cpu_fix_update(CFVMesh2D &mesh, CFVRecons2D &recons, CFVArray<double> &polution, double dt);

#endif // _H_KERNELS_CPU
