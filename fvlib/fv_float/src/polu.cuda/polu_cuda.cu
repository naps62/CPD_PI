#include <cuda.h>

#include "CUDA/CFVLib.h"
#include "FVLib.h"

#include "polu_cuda.h"

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
		fv_float dc) {

	// get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= num_edges) return;

	unsigned int i_left		= edge_left_cells[tid];
	unsigned int i_right	= edge_right_cells[tid];

	fv_float v_left[2], v_right[2];
	fv_float p_left, p_right;

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

	fv_float v	= ((v_left[0] + v_right[0]) * 0.5 * edge_normals_x[tid])
				+ ((v_left[1] + v_right[1]) * 0.5 * edge_normals_y[tid]);

	if (v < 0)
		flux[tid] = v * p_right;
	else
		flux[tid] = v * p_left;

	vs[tid] = v;
}

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
		fv_float dc) {

	fv_float result_vs;


	int threads_per_block = 512;

	dim3 num_blocks(num_edges % 512,1,1);
	dim3 num_threads(threads_per_block,1,1);
	
	cout << "running cuda_compute_flux<<<" << num_blocks.x << ", " << num_threads.x << ">>>" << endl;
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

	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		// something's gone wrong
		// print out the CUDA error as a string
		cout << "CUDA Error: " << cudaGetErrorString(error) << endl;

		// we can't recover from the error -- exit the program
		exit(-1);
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
		CudaFV::CFVVect<fv_float> &polution,
		CudaFV::CFVVect<fv_float> &flux,
		fv_float dt) {
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
		fv_float final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		fv_float mesh_parameter,
		FVVect<fv_float> &old_polution,
		CudaFV::CFVVect<fv_float> &polution,
		CudaFV::CFVPoints2D &velocities,
		CudaFV::CFVVect<fv_float> &flux,
		fv_float dc) {

	fv_float dt;
	fv_float t = 0;
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
	CudaFV::CFVVect<fv_float> vs(mesh.num_edges);
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

		FVLog::logger << "i\t" << "vs\t" << "flux" << endl;
		for(int j = 0; j < 30; ++j)
			FVLog::logger << j << "\t" << vs[j] << "\t" << flux[j] << endl;
		exit(0);

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
