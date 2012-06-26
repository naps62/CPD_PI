#ifndef _H_KERNELS_CPU
#define _H_KERNELS_CPU

#include "FVL/FVGlobal.h"
#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"

using namespace FVL;

/**
 * compute_mesh_parameter
 *
 * \param mesh
 */
double cpu_compute_mesh_parameter(CFVMesh2D &mesh);

/**
 * compute_edge_velocities
 *
 * \param mesh
 * \param velocities
 * \param vs
 * \param v_max
 */
void cpu_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max);

/**
 * compute_length_area_ratio
 *
 * \param mesh
 * \param length_area_ratio
 */
void cpu_compute_length_area_ratio(CFVMesh2D &mesh, CFVMat<double> &length_area_ratio);

/**
 * cpu_compute_flux
 *
 * \param mesh
 * \param velocity
 * \param polution
 * \param flux
 * \param dc
 */
void cpu_compute_flux(CFVMesh2D &mesh, CFVArray<double> &velocity, CFVArray<double> &polution, CFVArray<double> &flux, double dc);

/**
 * cpu_update
 *
 * \param mesh
 * \param polution
 * \param flux
 * \param dt
 */
void cpu_update(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &flux, double dt);
void cpu_update_optim(CFVMesh2D &mesh, CFVArray<double> &polution, CFVArray<double> &flux, double dt, CFVMat<double> &length_area_ratio);

#endif // _H_KERNELS_CPU
