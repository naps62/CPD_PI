#ifndef _CUH_KERNELS_CUDA
#define _CUH_KERNELS_CUDA

#include "FVL/CFVMesh2D.h"
#include "FVL/CFVArray.h"
using namespace FVL;


__host__ void cudaSafe(cudaError_t error, const string msg);

__host__ void cudaCheckError(const string msg);

__host__ double cpu_compute_mesh_parameter(CFVMesh2D &mesh);

__host__ void cpu_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max);

/* Compute matA inverse */
__global__
void kernel_compute_reverseA(
		CFVMesh2D_cuda *mesh,
		double **matA);

/* Compute system polution coeficients for system solve */
__global__
void kernel_compute_vecResult(
		CFVMesh2D_cuda *mesh,
		double *polution,
		double **vecResult,
		double dc);

__global__
void kernel_compute_vecABC(
		unsigned int num_cells,
		double **matA,
		double **vecResult,
		double **vecABC);

/* Compute flux */
__global__
void kernel_compute_flux(
		CFVMesh2D_cuda *mesh,
		double *polution,
		double *velocity,
		double **vecABC,
		double *flux,
		double dc);

/* polution update */
__global__
void kernel_update(
		CFVMesh2D_cuda *mesh,
		double *polution,
		double *flux,
		double dt);


/* Reduction of temporary velocities array */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__
void kernel_velocities_reduction(T *g_idata, T *g_odata, unsigned int n);

bool ispow2(unsigned int x);
unsigned int nextPow2(unsigned int x);
void get_reduction_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);
template<class T>
void wrapper_reduce_velocities(int size, int threads, int blocks, T *d_idata, T *d_odata);


#endif // _CUH_KERNELS_CUDA
