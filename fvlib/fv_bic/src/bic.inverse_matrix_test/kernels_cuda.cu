#include "kernels_cuda.cuh"

__global__
void kernel_compute_reverseA(
		unsigned int num_cells,
		double *cell_centroids_x,
		double *cell_centroids_y,
		unsigned int *cell_edges_count,
		unsigned int ** cell_edges,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double **matA) {

	// get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_cells) return;

	double x = cell_centroids_x[tid];
	double y = cell_centroids_y[tid];

	matA[0][tid] = x * x;	// elem (0, 0, tid)
	matA[1][tid] = x * y;	// elem (1, 0, tid)
	matA[2][tid] = x;		// elem (2, 0, tid)

	matA[3][tid] = x * y;	// elem (0, 1, tid)
	matA[4][tid] = y * y;	// elem (1, 1, tid)
	matA[5][tid] = y;		// elem (2, 1, tid)

	matA[6][tid] = x;		// elem (0, 2, tid)
	matA[7][tid] = y;		// elem (1, 2, tid)
	matA[8][tid] = 4;		// elem (2, 2, tid)

	unsigned int edge_limit = cell_edges_count[tid];
	for(unsigned int j = 0; j < edge_limit; ++j) {
		// get current edge
		unsigned int edge = cell_edges[j][tid];

		// get left cell of this edge
		unsigned int cell_j = edge_right_cells[edge];
		// if right cell is current one...
		if (cell_j == tid || cell_j == NO_RIGHT_CELL)
			cell_j = edge_left_cells[edge];

		//TODO: was the 2 factor forgotten in the formulas?
		x = cell_centroids_x[cell_j];
		y = cell_centroids_y[cell_j];

		matA[0][tid] += x * x;
		matA[1][tid] += x * y;
		matA[2][tid] += x;

		matA[3][tid] += x * y;
		matA[4][tid] += y * y;
		matA[5][tid] += y;

		matA[6][tid] += x;
		matA[7][tid] += y;
	}

	double det =	matA[0][tid] * (matA[4][tid] * matA[8][tid] -
									matA[7][tid] * matA[5][tid])
				-	matA[1][tid] * (matA[3][tid] * matA[8][tid] -
									matA[6][tid] * matA[5][tid])
				+	matA[2][tid] * (matA[3][tid] * matA[7][tid] -
									matA[6][tid] * matA[4][tid]);
	double invDet = 1.0 / det;

	double tmpA[9];
	tmpA[0] = (matA[4][tid] * matA[8][tid] - matA[7][tid] * matA[5][tid]) * invDet;
	tmpA[1] = (matA[3][tid] * matA[8][tid] - matA[6][tid] * matA[5][tid]) * invDet;
	tmpA[2] = (matA[3][tid] * matA[7][tid] - matA[6][tid] * matA[4][tid]) * invDet;

	tmpA[3] = (matA[1][tid] * matA[8][tid] - matA[7][tid] * matA[2][tid]) * invDet;
	tmpA[4] = (matA[0][tid] * matA[8][tid] - matA[6][tid] * matA[2][tid]) * invDet;
	tmpA[5] = (matA[0][tid] * matA[7][tid] - matA[6][tid] * matA[1][tid]) * invDet;

	tmpA[6] = (matA[1][tid] * matA[5][tid] - matA[4][tid] * matA[2][tid]) * invDet;
	tmpA[7] = (matA[0][tid] * matA[5][tid] - matA[3][tid] * matA[2][tid]) * invDet;
	tmpA[8] = (matA[0][tid] * matA[4][tid] - matA[3][tid] * matA[1][tid]) * invDet;

	for(unsigned int j = 0; j < 9; ++j)
		matA[j][tid] = tmpA[j];
}

/* Compute vecABC */
__global__
void kernel_compute_vecABC(
		unsigned int num_cells,
		double **matA,
		double **vecResult,
		double **vecABC) {

	// get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_cells) return;

	vecABC[0][tid]	= matA[0][tid] * vecResult[0][tid]
					+ matA[3][tid] * vecResult[1][tid]
					+ matA[6][tid] * vecResult[2][tid];

	vecABC[1][tid]	= matA[1][tid] * vecResult[0][tid]
					+ matA[4][tid] * vecResult[1][tid]
					+ matA[7][tid] * vecResult[2][tid];

	vecABC[2][tid]	= matA[2][tid] * vecResult[0][tid]
					+ matA[5][tid] * vecResult[1][tid]
					+ matA[8][tid] * vecResult[2][tid];
}

/* Compute system polution coeficients for system solve */
__global__
void kernel_compute_vecResult(
		unsigned int num_cells,
		double *cell_centroids_x,
		double *cell_centroids_y,
		unsigned int *cell_edges_count,
		unsigned int **cell_edges,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *polution,
		double **vecResult) {

	// get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_cells) return;

	double x = cell_centroids_x[tid];
	double y = cell_centroids_y[tid];
	double u = polution[tid];

	vecResult[0][tid] = u * x;
	vecResult[1][tid] = u * y;
	vecResult[2][tid] = u;

	unsigned int edge_limit = cell_edges_count[tid];
	for(unsigned int j = 0; j < edge_limit; ++j) {
		// get edge
		unsigned int edge = cell_edges[j][tid];

		// get right neighbor
		unsigned int neighbor = edge_right_cells[edge];

		// TODO ghost cell?
		if (neighbor == tid || neighbor == NO_RIGHT_CELL)
			neighbor = edge_left_cells[edge];

		x = cell_centroids_x[neighbor];
		y = cell_centroids_y[neighbor];
		u = polution[neighbor];

		vecResult[0][tid] += u * x;
		vecResult[1][tid] += u * y;
		vecResult[2][tid] += u;
		
	}
}

__global__
void kernel_compute_flux(
		unsigned int num_edges,
		//double *edge_normals_x,
		//double *edge_normals_y,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *edge_centroids_x,
		double *edge_centroids_y,
		double *polution,
		//double *velocity_x,
		//double *velocity_y,
		double *velocity,
		double **vecABC,
		double *flux,
		//double *vs,
		double dc) {

	// get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= num_edges) return;

	unsigned int i_left		= edge_left_cells[tid];
	unsigned int i_right	= edge_right_cells[tid];

	double p_left, p_right;
	double v;

	p_left		= polution[i_left];
	v			= velocity[tid];

	if (i_right != NO_RIGHT_CELL) {
		p_right	 	= polution[i_right];
	} else {
		p_right		= dc;
	}

	double x = edge_centroids_x[tid];
	double y = edge_centroids_y[tid];
	// TODupdate aqui
	double system_result = vecABC[0][tid] * x + vecABC[1][tid] * y + vecABC[2][tid];

	if (v < 0)
		flux[tid] = v * p_right;
	else
		flux[tid] = v * p_left;

	//vs[tid] = v;
}

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
		double dt) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_cells) return;

	// define start and end of neighbor edges
	//unsigned int* edge_index = cell_edges_index[tid];
	unsigned int edge_limit = cell_edges_count[tid];

	// get current polution value for this cell
	double new_polution	= 0;

	// for each edge of this cell
	for(unsigned int i = 0; i < edge_limit; ++i) {
		unsigned int edge = cell_edges[i][tid];
		// if this cell is at the left of the edge

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] *
			edge_lengths[edge] /
			cell_areas[tid];

		// if this cell is on the left or the right of the edge
		if (edge_left_cells[edge] == tid) {
			new_polution -= aux;
		} else {
			new_polution += aux;
		}
	}

	// update global value
	polution[tid] += new_polution;
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
	if (blockSize >= 64)  { if (tid <  32)  { if (smem[tid+ 32] > myMax) { smem[tid]  = myMax = smem[tid+  32]; } __syncthreads(); } }
	if (blockSize >= 32)  { if (tid <  16)  { if (smem[tid+ 16] > myMax) { smem[tid]  = myMax = smem[tid+  16]; } __syncthreads(); } }
	if (blockSize >= 32)  { if (tid <   8)  { if (smem[tid+  8] > myMax) { smem[tid]  = myMax = smem[tid+   8]; } __syncthreads(); } }
	if (blockSize >= 32)  { if (tid <   4)  { if (smem[tid+  4] > myMax) { smem[tid]  = myMax = smem[tid+   4]; } __syncthreads(); } }
	if (blockSize >= 32)  { if (tid <   2)  { if (smem[tid+  2] > myMax) { smem[tid]  = myMax = smem[tid+   2]; } __syncthreads(); } }
	if (blockSize >= 32)  { if (tid <   1)  { if (smem[tid+  1] > myMax) { smem[tid]  = myMax = smem[tid+   1]; } __syncthreads(); } }
	//if (blockSize >= 32) { if (smem[tid+16] > myMax) { smem[tid] = myMax = smem[tid+16]; } }
	//if (blockSize >= 16) { if (smem[tid+ 8] > myMax) { smem[tid] = myMax = smem[tid+ 8]; } }
	//if (blockSize >=  8) { if (smem[tid+ 4] > myMax) { smem[tid] = myMax = smem[tid+ 4]; } }
	//if (blockSize >=  4) { if (smem[tid+ 2] > myMax) { smem[tid] = myMax = smem[tid+ 2]; } }
	//if (blockSize >=  2) { if (smem[tid+ 1] > myMax) { smem[tid] = myMax = smem[tid+ 1]; } }

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

bool ispow2(unsigned int x) {
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

bool isPow2(unsigned int x) {
	return ((x & (x-1)) == 0);
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
	//cout << "shared size: " << smemSize << endl;

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


