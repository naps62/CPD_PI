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

template<class T>
struct SharedMemory {
	__device__ inline operator T*() {
		extern __shared__ int __smem[];
		return (T*)__smem;
	}
	__device__ inline operator const T*() const {
		extern __shared__ int __smem[];
		return (T*)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double> {
	__device__ inline operator double*() {
		extern __shared__ double __smem_d[];
		return (double*)__smem_d;
	}
	__device__ inline operator const double*() const {
		extern __shared__ double __smem_d[];
		return (double*)__smem_d;
	}
};

/**
 * TODO
 * reduction - still most naive implementation
 */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__
void kernel_velocities_reduction(T *g_idata, T *g_odata, unsigned int n) {

	T *sdata = SharedMemory<T>();

	// perform first level of reduction
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2 * gridDim.x;

	T myMax = g_idata[i];

	// we reduce multiple elements per thread. The number is determined by the
	// number of active thread blocks (via gridDim). More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while(i < n) {
		if (g_idata[i] > myMax) myMax = g_idata[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			if (g_idata[i+blockSize]) myMax = g_idata[i+blockSize];
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myMax;
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512) { if (tid < 256) { if (sdata[tid+256] > myMax) { sdata[tid] = myMax = sdata[tid+256]; } __syncthreads(); } }
	if (blockSize >= 256) { if (tid < 128) { if (sdata[tid+128] > myMax) { sdata[tid] = myMax = sdata[tid+128]; } __syncthreads(); } }
	if (blockSize >= 128) { if (tid <  64) { if (sdata[tid+ 64] > myMax) { sdata[tid] = myMax = sdata[tid+ 64]; } __syncthreads(); } }

		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and indice incorrect behavior.
		volatile T* smem = sdata;
	if (blockSize >= 64) { if (smem[tid+32] > myMax) { smem[tid] = myMax = smem[tid+32]; } }
	if (blockSize >= 32) { if (smem[tid+16] > myMax) { smem[tid] = myMax = smem[tid+16]; } }
	if (blockSize >= 16) { if (smem[tid+ 8] > myMax) { smem[tid] = myMax = smem[tid+ 8]; } }
	if (blockSize >=  8) { if (smem[tid+ 4] > myMax) { smem[tid] = myMax = smem[tid+ 4]; } }
	if (blockSize >=  4) { if (smem[tid+ 2] > myMax) { smem[tid] = myMax = smem[tid+ 2]; } }
	if (blockSize >=  2) { if (smem[tid+ 1] > myMax) { smem[tid] = myMax = smem[tid+ 1]; } }

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

bool isPow2(unsigned int x) {
	return ((x & (x-1)) == 0);
}

unsigned int nextPow2(unsigned int x) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

void get_reduction_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads) {
	threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
	blocks =  (n + (threads * 2- 1)) / (threads * 2);

	// TODO this was deleted. make sure it is safe
	//blocks = MIN(maxBlocks, blVocks);
}

template<class T>
void wrapper_reduce_velocities(int size, int threads, int blocks, T *d_idata, T *d_odata) {
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	if (isPow2(size)) {
		switch(threads) {
			case 512: kernel_velocities_reduction<T, 512, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case 256: kernel_velocities_reduction<T, 256, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case 128: kernel_velocities_reduction<T, 128, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case  64: kernel_velocities_reduction<T,  64, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case  32: kernel_velocities_reduction<T,  32, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case  16: kernel_velocities_reduction<T,  16, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case   8: kernel_velocities_reduction<T,   8, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case   4: kernel_velocities_reduction<T,   4, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case   2: kernel_velocities_reduction<T,   2, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case   1: kernel_velocities_reduction<T,   1, true><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
		}
	}
	else {
		switch(threads) {
			case 512: kernel_velocities_reduction<T, 512, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case 256: kernel_velocities_reduction<T, 256, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case 128: kernel_velocities_reduction<T, 128, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case  64: kernel_velocities_reduction<T,  64, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case  32: kernel_velocities_reduction<T,  32, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case  16: kernel_velocities_reduction<T,  16, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case   8: kernel_velocities_reduction<T,   8, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case   4: kernel_velocities_reduction<T,   4, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case   2: kernel_velocities_reduction<T,   2, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
			case   1: kernel_velocities_reduction<T,   1, false><<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); break;
		}
	}
}

// Instantiate reduction function for 3 types
template void wrapper_reduce_velocities<double>(int size, int threads, int blocks, double *d_idata, double *d_odata);
template void wrapper_reduce_velocities<float> (int size, int threads, int blocks, float  *d_idata, float  *d_odata);
template void wrapper_reduce_velocities<int>   (int size, int threads, int blocks, int    *d_idata, int    *d_odata);
/*void kernel_velocities_reduction(
		unsigned int n,
		int *g_input,
		int *g_output) {

	//int *sdata = SharedMemory();

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n) return;

	__syncthreads();

	for(unsigned int s=1; s < blockDim.x && i + s < n; s*=2) {
		if ((tid % (2*s)) == 0) {
			if (g_input[i + s] > g_input[i])
			g_input[i] = g_input[i + s];
		}
		__syncthreads();
	}
	__syncthreads();

	if (tid == 0) g_output[blockIdx.x] = g_input[i];
}*/

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
