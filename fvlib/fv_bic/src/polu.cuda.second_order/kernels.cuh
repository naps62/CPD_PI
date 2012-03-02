#ifndef _CUH_KERNELS_
#define _CUH_KERNELS_

#include "FVL/CFVMesh2D.h"
#include "FVL/CFVVect.h"
using namespace FVL;

/* Compute matA inverse */
#ifndef NO_CUDA
__global__
void kernel_compute_reverseA(
		//TODO
		);
#else
void cpu_compute_reverseA(
		CFVMesh2D &mesh,
		CFVMat<double> matA);
#endif

/* Compute flux */
#ifndef NO_CUDA
__global__
void kernel_compute_flux(
		unsigned int num_edges,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *polution,
		double *velocity,
		double *flux,
		double dc);
#else
void kernel_compute_flux(
		CFVMesh2D &mesh,
		CFVVect<double> &polution,
		CFVVect<double> &velocity,
		CFVVect<double> &flux,
		double dc);
#endif


/* polution update */
#ifndef NO_CUDA
__global__
void kernel_update(
		unsigned int num_cells,
		//unsigned int num_total_edges,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *edge_lengths,
		double *cell_areas,
		unsigned int **cell_edges,
		//unsigned int *cell_edges_index,
		unsigned int *cell_edges_count,
		double *polution,
		double *flux,
		double dt);
#else
void kernel_update(
		CFVMesh2D &mesh,
		CFVVect<double> &polution,
		CFVVect<double> &flux,
		double dt);
#endif

/* Reduction of temporary velocities array */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__
void kernel_velocities_reduction(T *g_idata, T *g_odata, unsigned int n);

bool ispow2(unsigned int x);
unsigned int nextPow2(unsigned int x);
void get_reduction_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);
template<class T>
void wrapper_reduce_velocities(int size, int threads, int blocks, T *d_idata, T *d_odata);


#endif // _CUH_KERNELS_
