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




__host__ void cuda_main_loop(
		double final_time,
		unsigned jump_interval,
		CudaFV::CFVMesh2D &mesh,
		double mesh_parameter,
		FVVect<double> &old_polution,
		CudaFV::CFVVect<double> &polutions,
		CudaFV::CFVPoints2D &velocities, CudaFV::CFVVect<double> &flux,
		double dc) {

	double t, dt;
	int i;
	FVio polution_file
}
