#include <cuda.h>

#include "CUDA/CFVLib.h"
#include "FVLib.h"
#include "CUDA/CFVProfile.h"

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

/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
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

	CudaFV::CFVProfile p_full_func("cuda_full_func");
	CudaFV::CFVProfile p_malloc("cuda_mallocs");
	CudaFV::CFVProfile p_main_loop("main_loop");
	CudaFV::CFVProfile p_compute_flux("compute_flux");
	CudaFV::CFVProfile p_reduction("reduction");
	CudaFV::CFVProfile p_update("update");
	CudaFV::CFVProfile p_output("file_output");

	PROF_START(p_full_func);

	// var declaration
	double dt;
	double t = 0;
	int i = 0;

	// open output file
	FVio polution_file("polution.xml", FVWRITE);
	polution_file.put(old_polution, t, "polution");
	
	PROF_START(p_malloc);

	// alloc space on device and copy data
	mesh.edge_normals.x.cuda_mallocAndSave();
	mesh.edge_normals.y.cuda_mallocAndSave();
	mesh.edge_lengths.cuda_mallocAndSave();
	mesh.edge_left_cells.cuda_mallocAndSave();
	mesh.edge_right_cells.cuda_mallocAndSave();
	mesh.cell_areas.cuda_mallocAndSave();
	mesh.cell_edges.cuda_mallocAndSave();
	mesh.cell_edges_index.cuda_mallocAndSave();
	mesh.cell_edges_count.cuda_mallocAndSave();

	polution.cuda_mallocAndSave();
	velocities.x.cuda_mallocAndSave();
	velocities.y.cuda_mallocAndSave();
	flux.cuda_malloc();

	PROF_STOP(p_malloc);

	// alloc space for tmp velocity vector
	CudaFV::CFVVect<double> vs(mesh.num_edges);
	vs.cuda_malloc();

	_DEBUG {
		unsigned int _d_copied_data_size =
				sizeof(double) * (
					mesh.edge_normals.x.size() + 
					mesh.edge_normals.y.size() +
					mesh.edge_lengths.size() +
					mesh.cell_areas.size() +
					polution.size() +
					velocities.x.size() +
					velocities.y.size() +
					flux.size()) +
				sizeof(unsigned int) * (
					mesh.edge_left_cells.size() +
					mesh.edge_right_cells.size() +
					mesh.cell_edges.size() +
					mesh.cell_edges_index.size() +
					mesh.cell_edges_count.size());
				
		unsigned int _d_full_data_size =
			_d_copied_data_size + sizeof(double) * (vs.size() + flux.size());

		float _d_transfer_rate =
			1000 * ((float) _d_copied_data_size / 1024 / p_malloc.getTime());
		
		CudaFV::CFVProfile::stream
			<< "Copied data size: " << (float) _d_copied_data_size / 1024 << " KB" << endl
			<< "Full data set size: " << (float) _d_full_data_size / 1024 << " KB" << endl
			<< "Data transfer rate: " << _d_transfer_rate << "KB/s" << endl;
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
	PROF_START(p_main_loop);
	while(t < final_time) {
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
				velocities.x.cuda_getArray(),
				velocities.y.cuda_getArray(),
				flux.cuda_getArray(),
				vs.cuda_getArray(),
				dc);
		PROF_STOP(p_compute_flux);

		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) {
			// something's gone wrong
			// print out the CUDA error as a string
			cout << "CUDA Error: " << cudaGetErrorString(error) << endl;
			// we can't recover from the error -- exit the program
			exit(-1);
		}

		/**
		 * Reduction of velocities
		 */
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
		dt = 1.0 / abs(max_vs) * mesh_parameter;

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
		if (i % jump_interval == 0) {
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
		}
	}
	PROF_STOP(p_main_loop);

	cout << "total iterations: " << i << endl;

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
	velocities.x.cuda_free();
	velocities.y.cuda_free();
	flux.cuda_free();

	PROF_STOP(p_full_func);
}

/**
 *
 * CUDA version must be higher than 1.3 (needs double-precision support)
 * Enough memory to run the kernels
 * Highest number of SMPs
 * Maximum of 8 GPUs
 */
/*__host__
int choseDevice(){
	int count, device = -1,  numSMP = 0, sel[8], flag = 0, flag2;
	cudaDeviceProp properties[8];
	// assumir doubles - ainda nao sei aquilo dos floats
	long int globMem = (30 + 16 * BLOCK_SIZE*GRID_SIZE + BLOCK_SIZE*GRID_SIZE)*8;

	cudaGetDeviceCount(&count);
	
	for(int i = 0; i < count && i < 8; i++){
		cudaGetDeviceProperties( &properties[i], i);
		if(properties[i].major >= 2){
			sel[i] = 1;
			flag++;
		}
		else
			sel[i] = 0;
		
		if(flag == 0){
			fprintf(flog, "There is no GPUs capable of running this program on your system.\n");
			return device;
		}

		fprintf(flog, "\nNumber of CUDA capable devices: %d\n", count);

		flag = 1;
		for(i = 0; i < count && i < 8; i++){
			if(sel[i]){
				if(properties[i].totalGlobalMem > globMem && properties[i].totalGlobalMem < 8000000000){
					sel[i] += flag;
					flag++;
					globMem = properties[i].totalGlobalMem;
				}
			}
		}
		
		flag2 = 1;

		for(i = 0; i < count && i < 8; i++){
			if(sel[i] == flag){
				if(properties[i].multiProcessorCount > numSMP){
					sel[i] += flag2;
					flag2++;
					numSMP = properties[i].multiProcessorCount;
				}
			}
		}

		for(i = 0; i < count && i < 8; i++){
			if(sel[i] == flag + flag2 - 1){
				cudaChooseDevice(&i, &properties[i]);
				cudaSetDevice(i);
				fprintf(flog, "\n%s GPU chosen\n", properties[i].name);
				fprintf(flog, "ID: %d\n", i);
				fprintf(flog, "Global Memory: %ld bytes\n", globMem);
				fprintf(flog, "Number of SMP: %d\n\n", numSMP);
				return i;
			}
		}
	}

	return device;
}*/
