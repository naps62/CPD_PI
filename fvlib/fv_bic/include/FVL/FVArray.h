/**
 * \file FVArray.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_FVARRAY
#define _H_FVARRAY

#include "FVL/FVLog.h"

namespace FVL {

	/**
	 * Generic array template class
	 */
	template<class T>
		class FVArray {
			public:
				unsigned int arr_size;	///< number of elems allocated
				T *arr;					///< Ptr to array memory

			public:
				/************************************************
				 * CONSTRUCTORS
				 ***********************************************/

				/**
				 * Constructor to create an array with a given size
				 *
				 * \param size Number of elements to allocate for the array
				 */
				FVArray(const unsigned int size);

				/**
				 * Constructor to create an array by copying an already existing array
				 *
				 * \param copy Array to copy
				 */
				FVArray(const FVArray<T> &copy);

				/**
				 * Default destructor
				 *
				 * Releases all memory allocated for this array
				 */
				~FVArray();

				/************************************************
				 * OPERATORS
				 ***********************************************/

				/**
				 * Array accessor as lvalue
				 *
				 * \param index position of the array to acess
				 * \return Modifiable reference to selected position of the array
				 */
				T &	operator [] (int index);

				/**
				 * Array accessor as rvalue
				 *
				 * Provides read-only acess to a given element
				 *
				 * \param index position of the array to access
				 * \return Constant reference to selected position of the array
				 */
				const T & operator [] (int index) const;

				/**
				 * Assignment operator
				 *
				 * Reallocate this array and create a hard copy of the parameter #FVArray
				 *
				 * \param copy #FVArray to copy
				 * \return Reference to the newly created #FVArray
				 */
				const FVArray<T> & operator = (const FVArray<T> & copy);

				/************************************************
				 * GETTERS/SETTERS
				 ***********************************************/

				/**
				 * Gives direct access to array memory
				 *
				 * \return Pointer to memory where array is allocated
				 */
				T* getArray();

				/**
				 * Gives size of the array (number of elements)
				 *
				 * \return total number of elements allocated
				 */
				unsigned int size() const;

			private:

				/************************************************
				 * MEMORY MANAGEMENT
				 ***********************************************/

				/**
				 * Allocate space for array
				 *
				 * \param size Number of elements to allocate
				 */
				void alloc(unsigned int size);

				/**
				 * Release allocated space for the array
				 */
				void dealloc();
		};

}

#include "FVL/templates/FVArray.hpp"

#endif // _H_CFVARRAY
