#include <cuda.h>

#include "FVLib.h"
#include "CUDA/CFVLib.h"
#include "FVLib.h"

#include "parameters.h"
#include "polu_cuda.h"
#include "kernels.cuh"


void gpu_update(
		CudaFV::CFVMesh2D &mesh,
		CudaFV::CFVVect<double> &polution,
		CudaFV::CFVVect<double> &flux,
		double dt) {

	// PARA CADA EDGE
	// 		POLUTION DA CELL A ESQUERDA: 
	for (unsigned int i = 0; i < mesh.num_edges; ++i) {
		polution[ (unsigned int) mesh.edge_left_cells[i] ] -=
			dt * flux[i] * mesh.edge_lengths[i] / mesh.cell_areas[ (unsigned int) mesh.edge_left_cells[i] ];
		if (mesh.edge_right_cells[i] != NO_RIGHT_EDGE)
			polution[ (unsigned int) mesh.edge_right_cells[i] ] +=
				dt * flux[i] * mesh.edge_lengths[i] / mesh.cell_areas[ (unsigned int) mesh.edge_right_cells[i] ];
	}
}

double compute_mesh_parameters (FVMesh2D mesh) {
	double h;
	double S;
	FVCell2D *cell;
	FVEdge2D *edge;

	h = 1.e20;
	for ( mesh.beginCell(); (cell = mesh.nextCell()) != NULL ; ) {
		S = cell->area;
		for ( cell->beginEdge(); (edge = cell->nextEdge()) != NULL; ) {
			if ( h * edge->length > S )
				h = S / edge->length;
		}
	}
	return h;
}

/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
int main() {

	CudaFV::CFVProfile p_full_func("cuda_full_func");
	CudaFV::CFVProfile p_malloc("cuda_mallocs");
	CudaFV::CFVProfile p_memcpy("cuda_memcpy");
	CudaFV::CFVProfile p_main_loop("main_loop");
	CudaFV::CFVProfile p_loop_iter("main_loop_iteration");
	CudaFV::CFVProfile p_compute_flux("compute_flux");
	CudaFV::CFVProfile p_reduction("reduction");
	CudaFV::CFVProfile p_update("update");
	CudaFV::CFVProfile p_output("file_output");

	PROF_START(p_full_func);

	string name;
	double h;
	double t;
	FVMesh2D old_mesh;

	Parameters data;
	data = read_parameters( "param.xml" );

	// read the mesh
	old_mesh.read( data.filenames.mesh.c_str() );

	// GPU
	CudaFV::CFVMesh2D mesh(mesh);

	FVVect<double> old_polution( old_mesh.getNbCell() );
	FVVect<double> old_flux( old_mesh.getNbEdge() );
	FVVect<FVPoint2D<double> > old_velocity( old_mesh.getNbCell() );

	//	read veloci
	FVio velocity_file( data.filenames.velocity.c_str() , FVREAD );
	velocity_file.get( old_velocity , t , name );

	FVio polu_ini_file( data.filenames.polution.initial.c_str() , FVREAD );
	polu_ini_file.get( old_polution , t , name );

	// GPU
	CudaFV::CFVVect<double> polution(mesh.num_cells);
	CudaFV::CFVVect<double> flux(mesh.num_edges);
	CudaFV::CFVPoints2D velocity(mesh.num_cells);

	for(unsigned int i = 0; i < polution.size(); ++i) {
		polution[i] = old_polution[i];
		velocity.x[i] = old_velocity[i].x;
		velocity.y[i] = old_velocity[i].y;
	}

	h = compute_mesh_parameters( old_mesh );

	// var declaration
	double dt;
	int i = 0;

	// open output file
	FVio polution_file("polution.xml", FVWRITE);
	polution_file.put(old_polution, t, "polution");
	
	PROF_START(p_malloc);
	// alloc space on device
	mesh.edge_normals.x.cuda_malloc();
	mesh.edge_normals.y.cuda_malloc();
	mesh.edge_lengths.cuda_malloc();
	mesh.edge_left_cells.cuda_malloc();
	mesh.edge_right_cells.cuda_malloc();
	mesh.cell_areas.cuda_malloc();
	mesh.cell_edges.cuda_malloc();
	mesh.cell_edges_index.cuda_malloc();
	mesh.cell_edges_count.cuda_malloc();
	polution.cuda_malloc();
	velocity.x.cuda_malloc();
	velocity.y.cuda_malloc();
	flux.cuda_malloc();
	// alloc space for tmp velocity vector
	CudaFV::CFVVect<double> vs(mesh.num_edges);
	vs.cuda_malloc();
	PROF_STOP(p_malloc);

	PROF_START(p_memcpy);
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	mesh.edge_normals.x.cuda_saveAsync(stream);
	mesh.edge_normals.y.cuda_saveAsync(stream);
	mesh.edge_lengths.cuda_saveAsync(stream);
	mesh.edge_left_cells.cuda_saveAsync(stream);
	mesh.edge_right_cells.cuda_saveAsync(stream);
	mesh.cell_areas.cuda_saveAsync(stream);
	mesh.cell_edges.cuda_saveAsync(stream);
	mesh.cell_edges_index.cuda_saveAsync(stream);
	mesh.cell_edges_count.cuda_saveAsync(stream);

	polution.cuda_saveAsync(stream);
	velocity.x.cuda_saveAsync(stream);
	velocity.y.cuda_saveAsync(stream);

	cudaStreamSynchronize(stream);
	PROF_STOP(p_memcpy);

	cudaStreamDestroy(stream);

	_DEBUG {
		float _d_copied_data_bytes =
				sizeof(double) * (
					mesh.edge_normals.x.size() + 
					mesh.edge_normals.y.size() +
					mesh.edge_lengths.size() +
					mesh.cell_areas.size() +
					polution.size() +
					velocity.x.size() +
					velocity.y.size() +
					flux.size()) +
				sizeof(unsigned int) * (
					mesh.edge_left_cells.size() +
					mesh.edge_right_cells.size() +
					mesh.cell_edges.size() +
					mesh.cell_edges_index.size() +
					mesh.cell_edges_count.size());

		float _d_copied_data_size=
			_d_copied_data_bytes / (float) 1024;
				
		float _d_full_data_bytes =
			_d_copied_data_bytes + sizeof(double) * (vs.size() + flux.size());

		float _d_full_data_size =
			_d_full_data_bytes / (float) 1024;

		float _d_transfer_rate =
			((float) _d_copied_data_size) * 1000 / p_memcpy.getTime();
		
		CudaFV::CFVProfile::stream
			<< "Copied data size: " << (float) _d_copied_data_size << " KB" << endl
			<< "Full data set size: " << (float) _d_full_data_size << " KB" << endl
			<< "Data transfer rate: " << _d_transfer_rate << " KB/s" << endl;
	}


	// prepare aux vars for reduction kernel
	int red_blocks, red_threads;
	get_reduction_num_blocks_and_threads(mesh.num_edges, 0, 512, red_blocks, red_threads);
	CudaFV::CFVVect<double> cpu_reducibles(red_blocks);
	cpu_reducibles.cuda_malloc();


	// select grid and block size for compute_flux kernel
	dim3 grid_size_cf(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_CF), 1, 1);
	dim3 block_size_cf(BLOCK_SIZE_CF, 1, 1);

	dim3 grid_size_up(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_UP), 1, 1);
	dim3 block_size_up(BLOCK_SIZE_UP, 1, 1);

	/**
	 * Beggining of main loop
	 */
	double final_time = data.time.final;
	double dc = data.computation.threshold;
	PROF_START(p_main_loop);
	while(t < final_time) {
		PROF_START(p_loop_iter);

		double max_vs;

		/**
		 * Invoke kernel for compute_flux
		 */
		PROF_START(p_compute_flux);
		kernel_compute_flux<<< grid_size_cf, block_size_cf >>>(
				mesh.num_edges,
				mesh.edge_normals.x.cuda_getArray(),
				mesh.edge_normals.y.cuda_getArray(),
				mesh.edge_left_cells.cuda_getArray(),
				mesh.edge_right_cells.cuda_getArray(),
				polution.cuda_getArray(),
				velocity.x.cuda_getArray(),
				velocity.y.cuda_getArray(),
				flux.cuda_getArray(),
				vs.cuda_getArray(),
				dc);
		PROF_STOP(p_compute_flux);

		/**
		 * Reduction of velocities
		 */

		vs.cuda_get();
		for(int i = 0; i vs.size(); ++i) {
			cout << vs[i] << endl;
		}
		exit(0);

		PROF_START(p_reduction);
		wrapper_reduce_velocities(mesh.num_edges, red_threads, red_blocks, vs.cuda_getArray(), cpu_reducibles.cuda_getArray());

		// reduce final array on cpu
		cpu_reducibles.cuda_get();
		max_vs = cpu_reducibles[0];
		for(unsigned int x = 1; x < cpu_reducibles.size(); ++x) {
			if (cpu_reducibles[x] > max_vs)
				max_vs = cpu_reducibles[x];
		}

		// based on max_vs, compute time elapsed
		dt = 1.0 / abs(max_vs) * h;

		PROF_STOP(p_reduction);

		/**
		 * Update polution values based on computed flux
		 */
		PROF_START(p_update);
		kernel_update<<< grid_size_up, block_size_up >>>(
				mesh.num_cells,
				mesh.num_total_edges,
				mesh.edge_left_cells.cuda_getArray(),
				mesh.edge_right_cells.cuda_getArray(),
				mesh.edge_lengths.cuda_getArray(),
				mesh.cell_areas.cuda_getArray(),
				mesh.cell_edges.cuda_getArray(),
				mesh.cell_edges_index.cuda_getArray(),
				mesh.cell_edges_count.cuda_getArray(),
				polution.cuda_getArray(),
				flux.cuda_getArray(),
				dt);
		PROF_STOP(p_update);

		t += dt;
		++i;

		/**
		 * Every <jump_interval> iterations, current polution values are saved to the output file polution.xml. with a low enough jump_interval, this creates an animated mesh of the polution along the entire time range, but also creates a bottleneck in the calculations
		 *
		 * Also, since the FVio class is still the original one (not updated to match the structs used for cuda), we first need to copy data to a structure of the old data types, and only then save it to file. This, again, has a big performance hit but is just temporary while the entire LIB is not CUDA-compatible
		 */
	/*	if (i % jump_interval == 0) {
			PROF_START(p_output);
			cout << "writing to file" << endl;
			polution.cuda_get();
			for(unsigned int x = 0; x < mesh.num_cells; ++x) {
				old_polution[x] = polution[x];
			}
			polution_file.put(old_polution, t, "polution");
			cout << "step " << i << " at time " << t << "\r";
			fflush(NULL);
			PROF_STOP(p_output);
		}*/

		PROF_STOP(p_loop_iter);
	}
	PROF_STOP(p_main_loop);

	FVLog::logger << "total iterations: " << i << endl;

	polution.cuda_get();
	for(unsigned int x = 0; x < mesh.num_cells; ++x) {
		old_polution[x] = polution[x];
	}
	polution_file.put(old_polution, t, "polution");

	// release memory on device
	mesh.edge_normals.x.cuda_free();
	mesh.edge_normals.y.cuda_free();
	mesh.edge_lengths.cuda_free();
	mesh.edge_left_cells.cuda_free();
	mesh.edge_right_cells.cuda_free();
	mesh.cell_areas.cuda_free();
	mesh.cell_edges.cuda_free();
	mesh.cell_edges_index.cuda_free();
	mesh.cell_edges_count.cuda_free();

	polution.cuda_free();
	velocity.x.cuda_free();
	velocity.y.cuda_free();
	flux.cuda_free();

	PROF_STOP(p_full_func);

	return 0;
}

