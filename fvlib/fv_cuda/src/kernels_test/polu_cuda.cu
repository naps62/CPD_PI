#include <cuda.h>

#include "polu_cuda.h"
#include "kernels.cuh"


/*
	Main loop: calculates the polution spread evolution in the time domain.
*/
int main() {

	int n = 10;
	int test[n];
	int result[3];
	int *d_test, *d_result;

	for(int i=0; i < n; ++i)
		test[i]=i;


	int blocks, threads;
	get_reduction_num_blocks_and_threads(n, 0, 256, blocks, threads);

	cudaMalloc(&d_test, sizeof(int)*8);
	cudaMemcpy(d_test, test, sizeof(int)*8, cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, sizeof(int)*2);
	cout << "before: " << endl;
	for(int i=0; i < n; ++i) {
		cout << test[i] << endl;
	}

	//kernel_velocities_reduction<<< numBlocks, numThreads >>>(11, d_test, d_result);
	wrapper_reduce_velocities(n, threads, blocks, d_test, d_result);

	cudaMemcpy(result, d_result, sizeof(int)*2, cudaMemcpyDeviceToHost);
	cout << "after: " << endl;
	for(int i=0; i < blocks; ++i) {
		cout << result[i] << endl;
	}
	exit(0);
}
