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

__global__ void cuda_compute_flux_kernel(
		unsigned int num_edges,
		unsigned int num_cells,
		double *edge_normals_x,
		double *edge_normals_y,
		double *edge_lengths,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *polution,
		double *flux ) {

	// get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// 
	if (tid >= num_edges) return;

	for(unsigned 
}

__host__ double cuda_compute_flux(
		unsigned int num_edges,
		unsigned int num_cells,
		double *edge_normals_x,
		double *edge_normals_y,
		double *edge_lengths,
		unsigned int *edge_left_cells,
		unsigned int *edge_right_cells,
		double *polution,
		double *flux,
		double *dt ) {

	double result_dt;

	cuda_compute_flux_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(num_edges, num_cells, edge_normals_x, edge_normals_y, edge_lengths, edge_left_cells, edge_right_cells, polution, flux);

	cudaMemcpy(&result_dt, dt, cudaMemcpyDeviceToHost);
	return result_dt;
}

__host__ void cuda_main_loop(
		double final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		double mesh_parameter,
		FVVect<double> &old_polution,
		CudaFV::CFVVect<double> &polution,
		CudaFV::CFVPoints2D &velocities,
		CudaFV::CFVVect<double> &flux,
		double dc) {

	double dt, *c_dt;
	double t = 0;
	int i = 0;
	
	FVio polution_file("polution.xml", FVWRITE);
	polution_file.put(old_polution, t, "polution");

	// alloc space on device and copy data
	mesh.edge_normals.x.cudaMallocAndSave();
	mesh.edge_normals.y.cudaMallocAndSave();
	mesh.edge_lengths.cudaMallocAndSave();
	mesh.edge_left_cells.cudaMallocAndSave();
	mesh.edge_right_cells.cudaMallocAndSave();

	polution.cudaMallocAndSave();
	velocities.cudaMallocAndSave();
	flux.cudaMalloc();

	cudaMalloc(&c_dt, sizeof(double));

	while(t < final_time) {
		// cuda_compute_flux will invoke the kernel and retrieve the calculated dt
		dt = cuda_compute_flux(
				mesh.num_edges,
				mesh.num_cells,
				mesh.edge_normals.x.getCudaArray(),
				mesh.edge_normals.y.getCudaArray(),
				mesh.edge_lengths.getCudaArray(),
				mesh.edge_left_cells.getCudaArray(),
				mesh.edge_right_cells.getCudaArray(),
				polution.getCudaArray(),
				velocities.getCudaArray(),
				flux.getCudaArray()
				c_dt);

		cout << "dt=" << dt << endl;
	}
}
