#include <cuda.h>

#include "polu_cuda.h"
#include "kernels.cuh"


/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
int main() {

	int test[8];
	int result[2];
	int *d_test, *d_result;

	for(int i=0; i < 8; ++i)
		test[i]=i;

	dim3 numBlocks=2;
	dim3 numThreads=4;

	cudaMalloc(&d_test, test, sizeof(int)*8);
	cudaMemcpy(d_test, test, sizeof(int)*8, cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, sizeof(int)*2);
	cout << "before: " << endl;
	for(int i=0; i < numBlocks.x; ++i) {
		cout << result[i] << endl;
	}
	kernel_velocities_reduction<<< numBlocks, numThreads >>>(256, test, d_result);
	cudaMemcpy(result, d_result, sizeof(int)*2, cudaMemcpyDeviceToHost);
	cout << "after: " << endl;
	for(int i=0; i < numBlocks.x; ++i) {
		cout << result[i] << endl;
	}
	exit(0);
}
