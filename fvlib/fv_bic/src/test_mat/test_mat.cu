#include "FVL/CFVMat.h"
using namespace FVL;

#include <vector>
#include <iostream>
using std::cout;
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void kernel_sum(int **X, int **Y, int **R, int w, int h) {
	unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int tidR = blockIdx.z + blockDim.z + threadIdx.z;

	int elem = tidY * w + tidX;
	int* matX = X[elem];
	int* matY = Y[elem];
	int* matR = R[elem];

	matR[tidR] = 0;//matX[tidR] + matY[tidR];
}

int main() {
	// array of 5 2x2 matrixes
	CFVMat<int> matX(2, 2, 5);
	CFVMat<int> matY(2, 2, 5);
	CFVMat<int> matR(2, 2, 5);

	for(unsigned int y = 0; y < 2; ++y)
		for(unsigned int x = 0; x < 2; ++x)
			for(unsigned int i = 0; i < 5; ++i)
				matX.elem(x, y, i) = matY.elem(x, y, i) = i;

	matX.cuda_mallocAndSave();
	matY.cuda_mallocAndSave();
	matR.cuda_malloc();

	dim3 gridDim = 1;
	dim3 blockDim(2, 2, 5);
	kernel_sum<<<gridDim, blockDim>>>(
			matX.cuda_getMat(),
			matY.cuda_getMat(),
			matR.cuda_getMat(),
			matX.width(),
			matY.height());

	matX.cuda_get();
	matR.cuda_get();
	for(unsigned int y = 0; y < 2; ++y) {
		for(unsigned int x = 0; x < 2; ++x) {
			cout << "coord: (" << x << ", " << y << ")";
			for(unsigned int i = 0; i < 5; ++i)
				cout << " " << matX.elem(x, y, i);
			cout << endl;
		}
	}
}
