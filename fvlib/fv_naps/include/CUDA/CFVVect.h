/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVPoints2D.h
 ** CUDA Vector (cuda-optimized storage, struct-of-arrays instead
 **    of array-of-structs
 **
 ** Author: Miguel Palhas, mpalhas@gmail.com
 ** -------------------------------------------------------------------------*/

#ifndef _CUDA_FVVECT
#define _CUDA_FVVECT

#include <cuda.h>
#include <cutil.h>

#include "MFVLog.h"

namespace CudaFV {

	template<class T>
		class CFVVect {
			private:
				unsigned int arr_size;
				T *arr;

				//T *cuda_arr;

			public:
				/**
				 * CONSTRUCTORS
				 */
				CFVVect();
				CFVVect(const unsigned int size);
				CFVVect(const CFVVect<T> &copy);
				~CFVVect();

				//~CFVVect() { dealloc(); }

				/**
				 * OPERATORS
				 */
				T & 				operator [] (int index);
				const T & 			operator []	(int index) const;
				const CFVVect<T> &	operator =	(const CFVVect<T> & copy);

				/**
				 * GETTERS/SETTERS
				 */
				T* getArray();
				unsigned int size() const;

				//T* cudaAlloc();
				//T* cudaGetPtr();
				//void cudaFree();
				//void cudaCopyToHost();
				//void cudaCopyToDevice();

			private:
				void alloc(unsigned int size);
				void dealloc();
				void test() {
				double *x;
					cudaMalloc(&x, sizeof(double)*2);
				}

		};

	/**
	 * CONSTRUCTORS
	 */
	template<class T>
		CFVVect<T>::CFVVect() {
			arr = NULL;
			alloc(0);
		}

	template<class T>
		CFVVect<T>::CFVVect(const unsigned int size) {
			alloc(size);
		}

	template<class T>
		CFVVect<T>::CFVVect(const CFVVect<T> &copy) {
			alloc(copy.size());
			for(unsigned int i = 0; i < arr_size; ++i) {
				arr[i] = copy[i];
			}
		}

	template<class T>
		CFVVect<T>::~CFVVect() {
			dealloc();
		}

	/**
	 * OPERATORS
	 */
	template<class T>
		T & CFVVect<T>::operator [] (int index) {
			return arr[index];
		}

	template<class T>
		const T & CFVVect<T>::operator [] (int index) const {
			return arr[index];
		}

	template<class T>
		const CFVVect<T> & CFVVect<T>::operator = (const CFVVect<T> & copy) {
			dealloc();
			alloc(copy.size());
			for(unsigned int i = 0; i < arr_size; ++i) {
				arr[i] = copy[i];
			}
			return *this;
		}

	template<class T>
		T* CFVVect<T>::getArray() {
			return arr;
		}

	template<class T>
		unsigned int CFVVect<T>::size() const {
			return arr_size;
		}

	/**
	 * CUDA
	 */
	/*template<class T>
		T* CFVVect<T>::cudaAlloc() {
			cudaMalloc(&cuda_arr, sizeof(double) * arr_size);
		}

	template<class T>
		T* CFVVect<T>::cudaGetPtr() {
			return cuda_arr;
		}

	template<class T>
		void CFVVect<T>::cudaFree() {
			cudaFree(cuda_arr);
		}

	template<class T>
		void CFVVect<T>::cudaCopyToHost() {
			cudaMemcpy(cuda_arr, arr, sizeof(double) * arr_size, cudaMemcpyHostToDevice);
		}

	template<class T>
		void CFVVect<T>::cudaCopyToDevice() {
			cudaMemcpy(arr, cuda_arr, sizeof(double) * arr_size, cudaMemcpyDeviceToHost);
		}*/

	/**
	 * ALLOC/DELETE
	 */
	template<class T>
		void CFVVect<T>::alloc(unsigned int size) {
			arr_size = size;
			if (arr_size > 0) {
				arr = new T[arr_size];
			}
		}

	template<class T>
		void CFVVect<T>::dealloc() {
			if (arr != NULL) {
				delete arr;
			}
			arr = NULL;
			arr_size = 0;
		}
}
#endif // _CUDA_FVVECT
