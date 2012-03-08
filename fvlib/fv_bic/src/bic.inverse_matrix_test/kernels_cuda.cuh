#ifndef _CUH_KERNELS_CUDA
#define _CUH_KERNELS_CUDA

#include "FVL/CFVMesh2D.h"
#include "FVL/CFVVect.h"
using namespace FVL;

/* Compute matA inverse */
__global__
void kernel_compute_reverseA(
		unsigned int num_cells,
		double *cell_centroids_x,
		double *cell_centroids_y,
		unsigned int *cell_edges_count,
		unsigned int **cell_edges,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double **matA);

/* Compute system polution coeficients for system solve */
__global__
void kernel_compute_vecResult(
		unsigned int num_cells,
		double *cell_centroids_x,
		double *cell_centroids_y,
		unsigned int *cell_edges_count,
		unsigned int **edge_edges,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *polution,
		double **vecResult);

__global__
void kernel_compute_vecABC(
		unsigned int num_cells,
		double **matA,
		double **vecResult,
		double **vecABC
		);

/* Compute flux */
__global__
void kernel_compute_flux(
		unsigned int num_edges,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *edge_centroids_x,
		double *edge_centroids_y,
		double *polution,
		double *velocity,
		double **vecABC,
		double *flux,
		double dc);


/* polution update */
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
