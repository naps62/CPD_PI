#include <cuda.h>
#include <cutil.h>

#include "CUDA/CFVLib.h"

__host__ void cuda_main_loop(
		double final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		double mesh_parameter,
		FVVect<double> &old_polution,
		CudaFV::CFVVect<double> &polutions,
		CudaFV::CFVPoints2D &velocities, CudaFV::CFVVect<double> &flux,
		double dc);

__global__ void cuda_compute_flux(...);
__global__ void cuda_reduce(...);
__global__ void cuda_update(...);



template<class T>
__host__ T * cuda_alloc_and_copy(CudaFV::CFVVect<T> &vec) {
	T* ptr;
	size_t size = sizeof(T) * vec.size();
	cudaMalloc(&ptr, size);
	cudaMemcpy(ptr, vec.getArray(), size, cudaMemcpyHostToDevice);
}

__host__ void cuda_main_loop(
		double final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		double mesh_parameter,
		FVVect<double> &old_polution,
		CudaFV::CFVVect<double> &polution,
		CudaFV::CFVPoints2D &velocities, CudaFV::CFVVect<double> &flux,
		double dc) {

	double dt;
	double t = 0;
	int i = 0;
	
	FVio polution_file("polution.xml", FVWRITE);
	polution_file.put(old_polution, t, "polution");

	// alloc space on device
	double *_edge_normals_x = cuda_alloc_and_copy(mesh.edge_normals.x);
	double *_edge_normals_y = cuda_alloc_and_copy(mesh.edge_normals.y);
	double *_edge_lengths	= cuda_alloc_and_copy(mesh.edge_lengths);
	int *_edge_left_cells	= cuda_alloc_and_copy(mesh.edge_left_cells);
	int *_edge_right_cells	= cuda_alloc_and_copy(mesh.edge_right_cells);
	double *_cell_areas		= cuda_alloc_and_copy(mesh.cell_areas);
	double *_polution		= cuda_alloc_and_copy(polution);
	double *_flux			= cuda_alloc_and_copy(flux);
	
	while(t < final_time) {
		
	}
}
