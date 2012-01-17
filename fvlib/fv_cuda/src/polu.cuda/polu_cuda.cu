#include <cuda.h>

#include "CUDA/CFVLib.h"
#include "FVLib.h"

#include "polu_cuda.h"

__global__
void cuda_compute_flux_kernel(
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

__host__
double cuda_compute_flux(
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
		CudaFV::CFVVect<double> &vs,
		double dc) {

	double result_vs;


	dim3 num_blocks(1,1,1);
	dim3 num_threads(16,1,1);
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
			vs.cuda_getArray(),
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

	vs.cuda_get();

	result_vs = vs[0];
	for(unsigned int i = 1; i < num_edges; ++i) {
		if (vs[i] > result_vs)
			result_vs = vs[i];
	}
	return 1.0 / abs(result_vs);
}

__host__
void gpu_update(
		CudaFV::CFVMesh2D &mesh,
		CudaFV::CFVVect<double> &polution,
		CudaFV::CFVVect<double> &flux,
		double dt) {
	for (unsigned int i = 0; i < mesh.num_edges; ++i) {
		polution[ (unsigned int) mesh.edge_left_cells[i] ] -=
			dt * flux[i] * mesh.edge_lengths[i] / mesh.cell_areas[ (unsigned int) mesh.edge_left_cells[i] ];
		if (mesh.edge_right_cells[i] != NO_RIGHT_EDGE)
			polution[ (unsigned int) mesh.edge_right_cells[i] ] +=
				dt * flux[i] * mesh.edge_lengths[i] / mesh.cell_areas[ (unsigned int) mesh.edge_right_cells[i] ];
	}
}

/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
__host__
void cuda_main_loop(
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
	int i = 0;
	
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
				vs,
				dc);

		dt *= mesh_parameter;

		flux.cuda_get();

		gpu_update(mesh, polution, flux, dt);
		t += dt;
		++i;

		if (i % jump_interval == 0) {
			for(unsigned int x = 0; x < mesh.num_cells; ++x) {
				old_polution[x] = polution[x];
			}
			polution_file.put(old_polution, t, "polution");
			cout << "step " << i << " at time " << t << "\r";
			fflush(NULL);
		}
	}

	for(unsigned int x = 0; x < mesh.num_cells; ++x) {
		old_polution[x] = polution[x];
	}
	polution_file.put(old_polution, t, "polution");

	mesh.edge_normals.x.cuda_free();
	mesh.edge_normals.y.cuda_free();
	mesh.edge_lengths.cuda_free();
	mesh.edge_left_cells.cuda_free();
	mesh.edge_right_cells.cuda_free();

	polution.cuda_free();
	velocities.x.cuda_free();
	velocities.y.cuda_free();
	flux.cuda_free();
}
