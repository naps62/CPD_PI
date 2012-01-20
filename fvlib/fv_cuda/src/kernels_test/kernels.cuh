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
__global__
void kernel_velocities_reduction(
		unsigned int n,
		int *g_input,
		int *g_output);

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
