#define OPTIM_LENGTH_AREA_RATIO
#define OPTIM_KERNELS

#include "../cuda.polu/main.cu"
#ifdef _CUDA
	#include "../cuda.polu/kernels_cuda.cu"
#else
	#include "../cuda.polu/kernels_cpu.cpp"
#endif