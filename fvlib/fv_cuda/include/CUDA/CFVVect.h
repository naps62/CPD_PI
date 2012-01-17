/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVVect.h
 ** CUDA Vector (cuda-optimized storage, struct-of-arrays instead
 **    of array-of-structs
 **
 ** Author: Miguel Palhas, mpalhas@gmail.com
 ** -------------------------------------------------------------------------*/

#ifndef _H_CUDA_FVVECT
#define _H_CUDA_FVVECT

#include "MFVLog.h"

namespace CudaFV {

	template<class T>
		class CFVVect {
			private:
				unsigned int arr_size;
				T *arr;
				T *cuda_arr;

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

				/**
				 * CUDA
				 */
				T* cuda_getArray();
				T* cuda_mallocAndSave();
				void cuda_malloc();
				void cuda_free();
				void cuda_save();
				void cuda_get();

			private:
				void alloc(unsigned int size);
				void dealloc();
		};

}

#include "CUDA/CFVVect.hpp"

#endif // _CUDA_FVVECT

