#include <cuda.h>

#include "CUDA/CFVLib.h"
#include "FVLib.h"

/*
 * Main loop: calculates the polution spread evolution in the time domain.
 */
__host__
void cuda_main_loop(
		fv_float final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		fv_float mesh_parameter,
		FVVect<fv_float> &old_polution,
		CudaFV::CFVVect<fv_float> &polutions,
		CudaFV::CFVPoints2D &velocities, CudaFV::CFVVect<fv_float> &flux,
		fv_float dc);

/*
 * compute flux (CUDA version)
 */
__host__
fv_float cuda_compute_flux(
		unsigned int num_edges,
		unsigned int num_cells,
		fv_float *edge_normals_x,
		fv_float *edge_normals_y,
		fv_float *edge_lengths,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		fv_float *polution,
		fv_float *velocities_x,
		fv_float *velocities_y,
		fv_float *flux,
		CudaFV::CFVVect<fv_float> &vs,
		fv_float dc);

/*
 * CUDA kernel for compute flux calc
 */
__global__
void cuda_compute_flux_kernel(
		unsigned int num_edges,
		unsigned int num_cells,
		fv_float *edge_normals_x,
		fv_float *edge_normals_y,
		fv_float *edge_lengths,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		fv_float *polution,
		fv_float *velocity_x,
		fv_float *velocity_y,
		fv_float *flux,
		fv_float *vs,
		fv_float dc);

/*
 * update function (still no CUDA implementation)
 */
__host__
void gpu_update(
		CudaFV::CFVMesh2D &mesh,
		CudaFV::CFVVect<fv_float> &polution,
		CudaFV::CFVVect<fv_float> &flux,
		fv_float dt);
