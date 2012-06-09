#ifndef _CUH_KERNELS_CUDA
#define _CUH_KERNELS_CUDA

#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
using namespace FVL;

/*******************
 * safety functions
 ******************/

/**
 * cudaSafe
 *
 * \param error
 * \param msg
 */
__host__ void cudaSafe(cudaError_t error, const string msg);

/**
 * cudaCheckError
 *
 * \param msg
 */
__host__ void cudaCheckError(const string msg);

/******************
 * functions still not using cuda
 *****************/

/**
 * cpu_compute_mesh_parameter
 *
 * \param mesh
 */
__host__ double kernel_compute_mesh_parameter(CFVMesh2D &mesh);

/**
 * cpu_compute_edge_velocities
 *
 * \param mesh
 * \param velocities
 * \param vs
 * \param v_max
 */
__host__ void kernel_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max);


/*****************
 * CUDA kernels
 ****************/

/**
 * kernel_compute_flux
 *
 * \param mesh
 * \param polution
 * \param velocity
 * \param flux
 * \param dc
 */
__global__ void kernel_compute_flux(CFVMesh2D_cuda *mesh, double *polution, double *velocity, double *flux, double dc);

/**
 * kernel_update
 *
 * \param mesh
 * \param polution
 * \param flux
 * \param dt
 */
__global__ void kernel_update1(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt);
__global__ void kernel_update2(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt);


// Reduction of temporary velocities array
template<class T, unsigned int blockSize, bool nIsPow2>
__global__
void kernel_velocities_reduction(T *g_idata, T *g_odata, unsigned int n);

bool ispow2(unsigned int x);
unsigned int nextPow2(unsigned int x);
void get_reduction_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);
template<class T>
void wrapper_reduce_velocities(int size, int threads, int blocks, T *d_idata, T *d_odata);


#endif // _CUH_KERNELS_CUDA
