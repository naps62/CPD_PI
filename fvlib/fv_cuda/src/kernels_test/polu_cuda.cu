#include <cuda.h>

#include "polu_cuda.h"
#include "kernels.cuh"


/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
int main() {

	int test[11];
	int result[3];
	int *d_test, *d_result;

	for(int i=0; i < 8; ++i)
		test[i]=i;
	test[8] = 10;
	test[9] = 11;
	test[10] = 12;

	result[0] = 62;
	result[1] = 25;

	dim3 numBlocks=3;
	dim3 numThreads=4;

	cudaMalloc(&d_test, sizeof(int)*8);
	cudaMemcpy(d_test, test, sizeof(int)*8, cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, sizeof(int)*2);
	cudaMemcpy(d_result, result, sizeof(int), cudaMemcpyHostToDevice);
	cout << "before: " << endl;
	for(int i=0; i < numBlocks.x; ++i) {
		cout << result[i] << endl;
	}

	kernel_velocities_reduction<<< numBlocks, numThreads >>>(11, d_test, d_result);

	cudaMemcpy(result, d_result, sizeof(int)*2, cudaMemcpyDeviceToHost);
	cout << "after: " << endl;
	for(int i=0; i < numBlocks.x; ++i) {
		cout << result[i] << endl;
	}
	exit(0);
}
