/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVVect.h
 ** CUDA Vector (cuda-optimized storage, struct-of-arrays instead
 **    of array-of-structs
 **
 ** Author: Miguel Palhas, mpalhas@gmail.com
 ** -------------------------------------------------------------------------*/

#ifndef _H_CFVVECT
#define _H_CFVVECT

#include <cuda.h>
#include <cuda_runtime.h>

#include "FVL/FVLog.h"

namespace FVL {

	template<class T>
		class CFVVect {
			public:
				unsigned int arr_size;
				T *arr;
				T *cuda_arr;

			public:
				/**
				 * CONSTRUCTORS
				 */
				CFVVect();
				CFVVect(const unsigned int size);
				CFVVect(const CFVVect<T> &copy);
				~CFVVect();

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

				/**
				 * CUDA
				 */
				T* cuda_getArray();
				T* cuda_mallocAndSave();
				void cuda_malloc();
				void cuda_free();
				void cuda_save();
				void cuda_saveAsync(cudaStream_t &stream);
				void cuda_get();

			protected:
				void alloc(unsigned int size);
				void dealloc();
		};

}

#include "FVL/CFVVect.hpp"

#endif // _H_CFVVECT

