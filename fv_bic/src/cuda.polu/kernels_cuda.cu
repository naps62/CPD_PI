#include "kernels_cuda.cuh"

__host__ void cudaSafe(cudaError_t error, const string msg) {
	if (error != cudaSuccess) {
		cerr << "Error: " << msg << " : " << error << endl;
		exit(-1);
	}
}

__host__ void cudaCheckError(const string msg) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		cerr << "Error: " << msg << " : " << cudaGetErrorString(error) << endl;
		exit(-1);
	}
}

// TODO: convert to cudaa
__host__  double cpu_compute_mesh_parameter(CFVMesh2D &mesh) {
	double h;
	double S;

	h = 1.e20;
	for(unsigned int cell = 0; cell < mesh.num_cells; ++cell) {
		S = mesh.cell_areas[cell];

		for(unsigned int edge = 0; edge < mesh.cell_edges_count[cell]; ++edge) {
			double length = mesh.edge_lengths[edge];
			if (h * length > S)
				h = S / length;
		}
	}

	return h;
}

__host__
void cpu_compute_edge_velocities(CFVMesh2D &mesh, CFVPoints2D<double> &velocities, CFVArray<double> &vs, double &v_max) {
	for(unsigned int i = 0; i < mesh.num_edges; ++i) {
		unsigned int left	= mesh.edge_left_cells[i];
		unsigned int right	= mesh.edge_right_cells[i];

		if (right == NO_RIGHT_CELL)
			right = left;

		double v	= ((velocities.x[left] + velocities.x[right]) * 0.5 * mesh.edge_normals.x[i])
					+ ((velocities.y[left] + velocities.y[right]) * 0.5 * mesh.edge_normals.y[i]);

		vs[i] = v;

		if (abs(v) > v_max || i == 0) {
			v_max = abs(v);
		}
	}
}

__global__
void kernel_compute_flux(CFVMesh2D_cuda *mesh, double *polution, double *velocity, double *flux, double dc) {
	// thread id = edge index
	unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (edge >= mesh->num_edges) return;

	// velocity of current edge
	double v = velocity[edge];

	if (v < 0)
		flux[edge] = v * polution[ mesh->edge_left_cells[edge] ];
	else
		flux[edge] = v * ((mesh->edge_right_cells[edge] == NO_RIGHT_CELL) ? dc : polution[ mesh->edge_right_cells[edge] ]);
}

__global__
void kernel_update(CFVMesh2D_cuda *mesh, double *polution, double *flux, double dt) {

	// thread id (cell index)
	unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x;

	// check boundaries
	if (cell >= num_cells) return;

	// define start and end of neighbor edges
	unsigned int edge_limit = mesh->cell_edges_count[cell];

	// get current polution value for this cell
	double new_polution	= 0;

	// for each edge of this cell
	for(unsigned int edge_i = 0; edge_i < edge_limit; ++i) {
		unsigned int edge = mesh->cell_edges[edge_i][cell];
		// if this cell is at the left of the edge

		// amount of polution transfered through the edge
		double aux = dt * flux[edge] *
			mesh->edge_lengths[edge] /
			mesh->cell_areas[cell];

		// if this cell is on the left or the right of the edge
		if (mesh->edge_left_cells[edge] == cell) {
			new_polution -= aux;
		} else {
			new_polution += aux;
		}
	}

	// update global value
	polution[cell] += new_polution;
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


