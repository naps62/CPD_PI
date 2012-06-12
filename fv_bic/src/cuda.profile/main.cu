#define PROFILE

#ifndef PROFILE_START
	#define PROFILE_INIT()
	#define PROFILE_CLEANUP()
#endif

#include "../cuda.polu/main.cu"
#ifdef _CUDA
	#include "../cuda.polu/kernels_cuda.cu"
#else
	#include "../cuda.polu/kernels_cpu.cpp"
#endif
