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

	result[0] = 62;
	result[1] = 25;

	dim3 numBlocks=2;
	dim3 numThreads=4;

	cudaMalloc(&d_test, sizeof(int)*8);
	cudaMemcpy(d_test, test, sizeof(int)*8, cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, sizeof(int)*2);
	cudaMemcpy(d_result, result, sizeof(int), cudaMemcpyHostToDevice);
	cout << "before: " << endl;
	for(int i=0; i < numBlocks.x; ++i) {
		cout << result[i] << endl;
	}

	int smemsize = numThreads.x * sizeof(int);
	if (numThreads.x < 32) smemsize *= 2;

	kernel_velocities_reduction<<< numBlocks, numThreads, smemsize >>>(8, test, d_result);

	cudaMemcpy(result, d_result, sizeof(int)*2, cudaMemcpyDeviceToHost);
	cout << "after: " << endl;
	for(int i=0; i < numBlocks.x; ++i) {
		cout << result[i] << endl;
	}
	exit(0);
}
