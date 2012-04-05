/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVMat.cpp
 ** CUDA Matrixes (cuda-optimized storage, struct-of-arrays instead
 **    of array-of-structs
 **
 ** Author:		Miguel Palhas, mpalhas@gmail.com
 ** Created:	13-02-2012
 ** Last Test:	---
 ** -------------------------------------------------------------------------*/

#ifdef _H_CFVMAT

#ifndef _HPP_CFVMAT
#define _HPP_CFVMAT

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace FVL {

	template<class T>
	T** CFVMat<T>::cuda_get() {
		return cuda_mat.cuda_get();
	}

	template<class T>
	T** CFVMat<T>::cuda_malloc() {
		// alloc an array of pointers to each elem
		for(unsigned int y = 0; y < h; ++y) {
			for(unsigned int x = 0; x < w; ++x) {
				cuda_mat[y * w + x] = mat[y * w + x].cuda_malloc();
			}
		}

		cuda_mat.cuda_mallocAndSave();
		return cuda_get();
	}

	template<class T>
	T** CFVMat<T>::cuda_mallocAndSave(cudaStream_t stream) {
		this->cuda_malloc();
		this->cuda_save(stream);
		return this->cuda_get();
	}

	template<class T>
	void CFVMat<T>::cuda_free() {
		cuda_mat.cuda_free();
		for(unsigned int y = 0; y < h; ++y)
			for(unsigned int x = 0; x < w; ++x)
				mat[y * w + x].cuda_free();
	}

	template<class T>
	void CFVMat<T>::cuda_save(cudaStream_t stream) {
		for(unsigned int y = 0; y < h; ++y)
			for(unsigned int x = 0; x < w; ++x)
				mat[y * w + x].cuda_save(stream);
	}

	template<class T>
	void CFVMat<T>::cuda_load(cudaStream_t stream) {
		for(unsigned int y = 0; y < h; ++y)
			for(unsigned int x = 0; x < w; ++x)
				mat[y * w + x].cuda_load(stream);
	}

	/* ALLOC/DELETE */
	template<class T>
	void CFVMat<T>::alloc(unsigned int w, unsigned int h, unsigned int size) {
		mat_size = size;
		this->w = w;
		this->h = h;
		if (mat_size > 0) {
			for(unsigned int y = 0; y < h; ++y)
				for(unsigned int x = 0; x < w; ++x)
					mat.push_back(CFVArray<T>(size));

			cuda_mat = CFVArray<T*>(w*h);
		}
	}

	template<class T>
	void CFVMat<T>::dealloc() {
		while(mat.size() > 0)
			mat.pop_back();
		mat_size = 0;
	}
}

#endif // _HPP_CFVMAT
#endif // _H_CFVMAT
