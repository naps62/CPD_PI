#include <cuda.h>
#include <cmath>

#include "FVL/CUDA/CFVLib.h"
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
		FVL::CFVMesh2D &mesh,
		double mesh_parameter,
		FVVect<double> &old_polution,
		FVL::CFVVect<double> &polutions,
		FVL::CFVPoints2D &velocities,
		FVL::CFVVect<double> &flux,
		double dc);

/*
 * cuda_min_reduction
 */

/*
 * update function (still no CUDA implementation)
 */
void gpu_update(
		FVL::CFVMesh2D &mesh,
		FVL::CFVVect<double> &polution,
		FVL::CFVVect<double> &flux,
		double dt);


/*__host__
int choseDevice();*/
