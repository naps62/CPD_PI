/**
 * \file CFVPoints2D.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_CUDA_FVPOINTS2D
#define _H_CUDA_FVPOINTS2D

#include "FVL/CFVArray.h"

namespace FVL {

	/**
	 * Generic CUDA-ready array of 2-dimensional pointers
	 *
	 * Stored with one array for x coord, and one for y
	 */
	template<class T>
		class CFVPoints2D {
			public:
				CFVArray<T> x, y;	///< Arrays for each coord

				/************************************************
				 * CONSTRUCTORS
				 ***********************************************/

				/**
				 * Constructor to create a CUDA-ready array of 2-dimensional pointers with a given size
				 *
				 * \param size Number of 2-dimensional elements to allocate for the array
				 */
				CFVPoints2D(const unsigned int size)            { x = CFVArray<T>(size);    y = CFVArray<T>(size); }

				/**
				 * Constructor to create a CUDA-ready array of 2-dimensional pointers by copying an already existing array
				 *
				 * \param copy Array to copy
				 */
				CFVPoints2D(const FVL::CFVPoints2D<T> &copy)    { x = CFVArray<T>(copy.x);  y = CFVArray<T>(copy.y); }

				/************************************************
				 * GETTERS/SETTERS
				 ***********************************************/

				/**
				 * Gives size of the array (number of (x, y) elements)
				 *
				 * \return Total number of elements allocated
				 */
				unsigned int size() {
					return x.size();
				}
		};
}

#endif // _H_CUDA_FVPOINTS2D

