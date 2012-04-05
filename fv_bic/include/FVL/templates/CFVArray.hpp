/**
 * \file CFVArray.hpp
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifdef _H_CFVARRAY

#ifndef _HPP_CFVARRAY
#define _HPP_CFVARRAY

namespace FVL {

	/************************************************
	 * CUDA
	 ***********************************************/

	template<class T>
		T* CFVArray<T>::cuda_get() {
			return cuda_arr;
		}

	template<class T>
		T* CFVArray<T>::cuda_mallocAndSave(cudaStream_t stream) {
			this->cuda_malloc();
			this->cuda_save(stream);
			return this->cuda_get();
		}

	template<class T>
		T* CFVArray<T>::cuda_malloc() {
			cudaMalloc((void **)&cuda_arr, sizeof(T) * this->arr_size);
			return this->cuda_get();
		}

	template<class T>
		void CFVArray<T>::cuda_free() {
			cudaFree(cuda_arr);
		}

	template<class T>
		void CFVArray<T>::cuda_save(cudaStream_t stream) {
			cudaMemcpyAsync(cuda_arr, this->arr, sizeof(T) * this->arr_size, cudaMemcpyHostToDevice, stream);
		}

	template<class T>
		void CFVArray<T>::cuda_load(cudaStream_t stream) {
			cudaMemcpy(this->arr, cuda_arr, sizeof(T) * this->arr_size, cudaMemcpyDeviceToHost);
		}

}

#endif // _HPP_CFVARRAY
#endif // _H_CFVARRAY

