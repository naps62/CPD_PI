#ifndef _CUH_KERNELS_CUDA
#define _CUH_KERNELS_CUDA

#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
using namespace FVL;


__host__ void cudaSafe(cudaError_t error, const string msg);

__host__ void cudaCheckError(const string msg);

__host__ double cpu_compute_mesh_parameter(CFVMesh2D &mesh);

__host__ void cpu_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max);

/**
 * Compute matA inverse
 *
 * \param mesh
 * \param matA
 */
__global__ void kernel_compute_reverseA(CFVMesh2D_cuda *mesh, double **matA);

/**
 * Compute system polution coeficients for system solve
 *
 * \param mesh
 * \param polution
 * \param vecResult
 * \param dc
 */
__global__ void kernel_compute_vecResult(CFVMesh2D_cuda *mesh, double *polution, double **vecResult, double dc);

/**
 * Compute value of the ABC vector
 * 
 * \param num_cells
 * \param matA
 * \param vecResult
 * \param vecABC
 */
__global__ void kernel_compute_vecABC(unsigned int num_cells, double **matA, double **vecResult, double **vecABC);

/**
 * Compute flux
 * 
 * \param mesh
 * \param polution
 * \param velocity
 * \param flux
 * \param dc
 */
__global__ void kernel_compute_flux(CFVMesh2D_cuda *mesh, double *polution, double *velocity, double **vecABC, double *flux, double dc, double t, double dt);

/**
 * polution update
 *
 * \param mesh
 * \param polution
 * \param flux
 * \param dt
 */
__global__ void kernel_update(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt);

/**
 * Reset oldflux array to 0
 *
 * \param mesh
 * \param oldflux
 */
__global__ void kernel_reset_oldflux(CFVMesh2D_cuda *mesh, double *oldflux);

/**
 * Detects overflow errors in computed polution values
 *
 * \param mesh
 * \param polution
 * \param flux
 * \param oldflux
 * \param invalidate_flux
 */
__global__ void kernel_detect_polution_errors(CFVMesh2D_cuda *mesh, double *polution, double *flux, double *oldflux, bool *invalidate_flux);

/**
 * Recalculate flux for cells where error was found
 *
 * \param mesh
 * \param polution
 * \param velocity
 * \param flux
 * \param oldflux
 * \param invalidate_flux
 */
__global__ void kernel_fix_polution_errors(CFVMesh2D_cuda *mesh, double *polution, double *velocity, double *flux, double *oldflux, bool *invalidate_flux);

/**
 * Re-updates polution values, based on difference between newly calculated fluxes and old ones
 *
 * \param mesh
 * \param polution
 * \param flux
 * \param oldflux
 * \param dt
 * \param invalidate_flux
 */
__global__ void kernel_fix_update(CFVMesh2D_cuda *mesh, double *polution, double *flux, double *oldflux, double dt, bool *invalidate_flux);

#endif // _CUH_KERNELS_CUDA
