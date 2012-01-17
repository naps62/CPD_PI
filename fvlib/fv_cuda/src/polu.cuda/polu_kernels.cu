#include <cuda.h>

#include "CUDA/CFVLib.h"
#include "FVLib.h"

__host__ void cuda_main_loop(
		double final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		double mesh_parameter,
		FVVect<double> &old_polution,
		CudaFV::CFVVect<double> &polutions,
		CudaFV::CFVPoints2D &velocities, CudaFV::CFVVect<double> &flux,
		double dc);

__global__ void cuda_compute_flux_kernel(
		unsigned int num_edges,
		unsigned int num_cells,
		double *edge_normals_x,
		double *edge_normals_y,
		double *edge_lengths,
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
	edge_left_cells[tid] = tid;
	edge_normals_x[tid] = edge_normals_y[tid];
	//polution[tid] = velocity_y[tid];
	return;
	if (tid >= num_edges) return;

	unsigned int i_left		= edge_left_cells[tid];
	unsigned int i_right	= edge_right_cells[tid];

	double v_left[2], v_right[2];
	double p_left, p_right;

	v_left[0]	= velocity_x[i_left];
	v_left[1]	= velocity_y[i_left];
	p_left		= polution[i_left];

	if (i_right == NO_RIGHT_EDGE) {
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

	vs[tid] = tid;
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
		double *velocities_x,
		double *velocities_y,
		double *flux,
		double *vs,
		double dc) {

	//double result_vs;


	dim3 num_blocks(1,1);
	dim3 num_threads(512,1);
	cuda_compute_flux_kernel<<<num_blocks, num_threads>>>(
			num_edges,
			num_cells,
			edge_normals_x,
			edge_normals_y,
			edge_lengths,
			edge_left_cells,
			edge_right_cells,
			polution,
			velocities_x,
			velocities_y,
			flux,
			vs,
			dc);

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		// something's gone wrong
		// print out the CUDA error as a string
		cout << "CUDA Error: " << cudaGetErrorString(error) << endl;

		// we can't recover from the error -- exit the program
		return 1;
	}

	//cudaMemcpy(&result_vs, vs, sizeof(double), cudaMemcpyDeviceToHost);
	//return 1.0 / abs(result_vs);
	return 0;
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

	double dt;
	double t = 0;
	
	FVio polution_file("polution.xml", FVWRITE);
	polution_file.put(old_polution, t, "polution");
	
	// alloc space on device and copy data
	mesh.edge_normals.x.cuda_mallocAndSave();
	mesh.edge_normals.y.cuda_mallocAndSave();
	mesh.edge_lengths.cuda_mallocAndSave();
	mesh.edge_left_cells.cuda_mallocAndSave();
	mesh.edge_right_cells.cuda_mallocAndSave();

	polution.cuda_mallocAndSave();
	velocities.x.cuda_mallocAndSave();
	velocities.y.cuda_mallocAndSave();
	flux.cuda_malloc();

	// alloc space for tmp velocity vector
	CudaFV::CFVVect<double> vs(mesh.num_edges);
	vs.cuda_malloc();

	while(t < final_time) {
		// cuda_compute_flux will invoke the kernel and retrieve the calculated dt
		dt = cuda_compute_flux(
				mesh.num_edges,
				mesh.num_cells,
				mesh.edge_normals.x.cuda_getArray(),
				mesh.edge_normals.y.cuda_getArray(),
				mesh.edge_lengths.cuda_getArray(),
				mesh.edge_left_cells.cuda_getArray(),
				mesh.edge_right_cells.cuda_getArray(),
				polution.cuda_getArray(),
				velocities.x.cuda_getArray(),
				velocities.y.cuda_getArray(),
				flux.cuda_getArray(),
				vs.cuda_getArray(),
				dc);


		vs.cuda_get();
		mesh.edge_normals.x.cuda_get();
		mesh.edge_normals.y.cuda_get();
		mesh.edge_lengths.cuda_get();
		mesh.edge_left_cells.cuda_get();
		mesh.edge_right_cells.cuda_get();
		polution.cuda_get();
		velocities.x.cuda_get();
		velocities.y.cuda_get();
		flux.cuda_get();
		//velocities.x.cuda_get();
		//mesh.edge_lengths.cuda_get();
		//mesh.edge_left_cells.cuda_get();

		for(unsigned int i=0; i < mesh.num_edges; ++i) {
			cout << "tid= " << mesh.edge_left_cells[i] << " vs= " << mesh.edge_normals.x[i] << " custom= " << mesh.edge_normals.y[i];
			getchar();
		}

	}
}
