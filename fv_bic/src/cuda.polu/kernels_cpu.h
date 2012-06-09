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

#endif // _H_KERNELS_CPU
