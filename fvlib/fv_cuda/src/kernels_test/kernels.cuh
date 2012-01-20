#ifndef _CUH_KERNELS_
#define _CUH_KERNELS_

/*
 * Compute flux calc
 */
__global__
void kernel_compute_flux(
		unsigned int num_edges,
		double *edge_normals_x,
		double *edge_normals_y,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *polution,
		double *velocity_x,
		double *velocity_y,
		double *flux,
		double *vs,
		double dc);

/**
 * Reduction of temporary velocities array
 */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__
void kernel_velocities_reduction(
		T *g_idata,
		T *g_odata,
		unsigned int n);

bool isPow2(unsigned int x);
unsigned int nextPow2(unsigned int x);
void get_reduction_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

template<class T>
void wrapper_reduce_velocities(int size, int threads, int blocks, T *d_idata, T *d_odata);

/**
 * polution update
 */
 __global__
 void kernel_update(
 		unsigned int num_edges,
 		unsigned int *edge_left_cells,
 		unsigned int *edge_right_cells,
 		double *lengths,
 		double *areas,
 		double *polution,
 		double *flux,
 		double dt);

#endif // _CUH_KERNELS_
