#include "kernels.cuh"

#include "CUDA/CFVLib.h"

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
		double dc) {

	// get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= num_edges) return;

	unsigned int i_left		= edge_left_cells[tid];
	unsigned int i_right	= edge_right_cells[tid];

	double v_left[2], v_right[2];
	double p_left, p_right;

	v_left[0]	= velocity_x[i_left];
	v_left[1]	= velocity_y[i_left];
	p_left		= polution[i_left];

	if (i_right != NO_RIGHT_EDGE) {
		v_right[0]	= velocity_x[i_right];
		v_right[1]	= velocity_y[i_right];
		p_right	 	= polution[i_right];
	} else {
		v_right[0]	= v_left[0];
		v_right[1]	= v_left[1];
		p_right		= dc;
	}

	double v	= ((v_left[0] + v_right[0]) * 0.5 * edge_normals_x[tid])
				+ ((v_left[1] + v_right[1]) * 0.5 * edge_normals_y[tid]);

	if (v < 0)
		flux[tid] = v * p_right;
	else
		flux[tid] = v * p_left;

	vs[tid] = v;
}

struct SharedMemory {
	__device__ inline operator int*() {
		extern __shared__ int __smem[];
		return (int*)__smem;
	}
	__device__ inline operator const int*() const {
		extern __shared__ int __smem[];
		return (int*)__smem;
	}
};

/**
 * TODO
 * reduction - still most naive implementation
 */
__global__
void kernel_velocities_reduction(
		unsigned int n,
		int *g_input,
		int *g_output) {

	g_output[0] = 1;
	int *sdata = SharedMemory();

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_input[i] : 0;

	__syncthreads();

	for(unsigned int s=1; s < blockDim.x; s*=2) {
		if ((tid % (2*s)) == 0) {
			if (sdata[tid + s] > sdata[tid])
			sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) g_output[blockIdx.x] = sdata[0];
}

__global__
void kernel_update(
		unsigned int num_edges,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *lengths,
		double *areas,
		double *polution,
		double *flux,
		double dt) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_edges) return;

	unsigned int left	= edge_left_cells[tid];
	unsigned int right	= edge_right_cells[tid];

	polution[left] =
		dt * flux[tid] * lengths[tid] / areas[left];

	if (right == NO_RIGHT_EDGE)
		return;

	polution[right] =
		dt * flux[tid] * lengths[tid] / areas[right];
}
