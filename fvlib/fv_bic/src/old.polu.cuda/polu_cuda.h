#include <cuda.h>
#include <cmath>

#include "CUDA/CFVLib.h"
#include "FVLib.h"

#define BLOCK_SIZE_CF	512
#define BLOCK_SIZE_RED	512
#define BLOCK_SIZE_UP	512
#define GRID_SIZE(elems, threads)	((int) std::ceil((double) elems / threads))

/*
 * Main loop: calculates the polution spread evolution in the time domain.
 */
void cuda_main_loop(
		double final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		double mesh_parameter,
		FVVect<double> &old_polution,
		CudaFV::CFVVect<double> &polutions,
		CudaFV::CFVPoints2D &velocities,
		CudaFV::CFVVect<double> &flux,
		double dc);

/*
 * cuda_min_reduction
 */

/*
 * update function (still no CUDA implementation)
 */
void gpu_update(
		CudaFV::CFVMesh2D &mesh,
		CudaFV::CFVVect<double> &polution,
		CudaFV::CFVVect<double> &flux,
		double dt);


/*__host__
int choseDevice();*/
