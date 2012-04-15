/**
 * \file CFVArray.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_CFVARRAY
#define _H_CFVARRAY

#include "FVL/FVLog.h"
#include "FVL/FVArray.h"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace FVL {

	/**
	 * Generic CUDA-ready array template class
	 */
	template<class T>
		class CFVArray : public FVArray<T> {
			public:
				T *cuda_arr;	///< Ptr to CUDA memory of the array (NULL if not allocated)

			public:
				/************************************************
				 * CONSTRUCTORS
				 ***********************************************/

				/**
				 * Empty constructor
				 */
				CFVArray() : FVArray<T>() { }

				/**
				 * Constructor to create a CUDA-ready array with a given size
				 *
				 * \param size Number of elements to allocate for the array
				 */
				CFVArray(const unsigned int size) : FVArray<T>(size) { }

				/**
				 * Constructor to create an array by copying an already existing array
				 *
				 * \param copy Array to copy
				 */
				CFVArray(const CFVArray<T> &copy) : FVArray<T>(copy) { }

				/**
				 * Default destructor
				 *
				 * Releases all memory allocated for this array
				 */
				~CFVArray() { }

				/************************************************
				 * CUDA
				 ***********************************************/
				#ifdef __CUDACC__

				/**
				 * Get pointer to CUDA memory of the array
				 *
				 * \return Pointer to allocated memory on CUDA device (NULL if not allocated)
				 */
				T* cuda_get();

				/**
				 * Allocates CUDA memory for the array, and copies data to it
				 *
				 * Equivalent to calling cuda_malloc() and cuda_save() in sequence.
				 *
				 * \param stream CUDA Stream to use (defaults to 0 to use no stream)
				 * \return Pointer to allocated memory on CUDA device (NULL in case of allocation error)
				 */
				T* cuda_mallocAndSave(cudaStream_t stream = 0);

				/**
				 * Allocates CUDA memory for the array
				 *
				 * \return Pointer to allocated memory on CUDA device (NULL in case of allocation error)
				 */
				T* cuda_malloc();

				/**
				 * Releases CUDA memory of this array
				 */
				void cuda_free();

				/**
				 * Copies data from the array on host memory to CUDA memory
				 *
				 * \param stream CUDA Stream to use (defaults to 0 to use no stream)
				 */
				void cuda_save(cudaStream_t stream = 0);

				/**
				 * Copies data from the array on CUDA memory back to the host
				 *
				 * \param stream CUDA Stream to use (defaults to 0 to use no stream)
				 */
				void cuda_load(cudaStream_t stream = 0);

				#endif // __CUDACC__
		};
}

#include "FVL/templates/CFVArray.hpp"

#endif // _H_CFVARRAY

