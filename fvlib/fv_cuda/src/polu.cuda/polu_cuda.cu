#include <cuda.h>

#include "CUDA/CFVLib.h"
#include "FVLib.h"

#include "polu_cuda.h"
#include "kernels.cuh"


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

	// var declaration
	double dt;
	double t = 0;
	int i = 0;

	CudaFV::CFVVect<int> test(256);
	int *result, *d_result;
	for(int i=0; i < 256; ++i)
		test[i]=i;

	test.cuda_mallocAndSave();
	dim3 numBlocks=4;
	dim3 numThreads=256/4;
	result = (int *) malloc(sizeof(int)*4);
	cudaMalloc(&d_result, sizeof(int)*numBlocks.x);
	kernel_velocities_reduction<<< numBlocks, numThreads >>>(
			256,
			test.cuda_getArray(),
			result);
	cudaMemcpy(result, d_result, sizeof(int)*4, cudaMemcpyDeviceToHost);
	cout << "reduction result: " << endl;
	for(int i=0; i < 4; ++i) {
		cout << result[i] << endl;
	}
	exit(0);


	
	// open output file
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

	// select grid and block size
	dim3 grid_size_cf(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_CF), 1, 1);
	dim3 block_size_cf(BLOCK_SIZE_CF, 1, 1);

	dim3 grid_size_red(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_RED), 1, 1);
	dim3 block_size_red(BLOCK_SIZE_RED, 1, 1);

	dim3 grid_size_up(GRID_SIZE(mesh.num_edges, BLOCK_SIZE_UP), 1, 1);
	dim3 block_size_up(BLOCK_SIZE_UP, 1, 1);

	/**
	 * Beggining of main loop
	 */
	while(t < final_time) {
		double max_vs;

		/**
		 * Invoke kernel for compute_flux
		 */
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
	/*	kernel_velocities_reduction<<< grid_size_red, block_size_red >>>(
				mesh.num_edges,
				vs.cuda_getArray());*/

		max_vs = cudaMemcpy(&max_vs, vs.cuda_getArray(), sizeof(double), cudaMemcpyDeviceToHost);
		dt = 1.0 / abs(max_vs) * mesh_parameter;

		/*cuda_update<<< grid_size_up, block_size_up >>>(
				mesh.num_edges,
				mesh.edge_left_cells.cuda_getArray(),
				mesh.edge_right_cells.cuda_getArray(),
				polution.cuda_getArray(),
				flux.cuda_getArray(),
				dt);*/


		/**
		 * update function is not yet implemented in CUDA. To invoke the C++ version, a cudaMemcpy is required before it to copy flux parameter, and after, to update polution value on the GPU
		 * Due to this, this implementation is not yet efficient compared to the original code
		 */
		flux.cuda_get();
		gpu_update(mesh, polution, flux, dt);
		polution.cuda_save();

		t += dt;
		++i;

		/**
		 * Every <jump_interval> iterations, current polution values are saved to the output file polution.xml. with a low enough jump_interval, this creates an animated mesh of the polution along the entire time range, but also creates a bottleneck in the calculations
		 *
		 * Also, since the FVio class is still the original one (not updated to match the structs used for cuda), we first need to copy data to a structure of the old data types, and only then save it to file. This, again, has a big performance hit but is just temporary while the entire LIB is not CUDA-compatible
		 */
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

	// release memory on device
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