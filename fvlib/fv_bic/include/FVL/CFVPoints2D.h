/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVPoints2D.h
 ** CUDA Arrayor of 2D Points (cuda-optimized storage, struct-of-arrays instead
 **    of array-of-structs)
 **
 ** Author:		Miguel Palhas, mpalhas@gmail.com
 ** Created:	13-02-2012
 ** Last Test:	---
 ** TODO: optimize storage; maybe implement as subclass of CFVMat instead of using 2 CFVArray
 ** -------------------------------------------------------------------------*/

#ifndef _H_CUDA_FVPOINTS2D
#define _H_CUDA_FVPOINTS2D

#include "FVL/CFVArray.h"

namespace FVL {

	template<class T>
		class CFVPoints2D {
			public:
				/**
				 * one array for X coord, another for Y
				 */
				CFVArray<T> x, y;

				CFVPoints2D()                                   { }
				CFVPoints2D(const unsigned int size)            { x = CFVArray<T>(size);    y = CFVArray<T>(size); }
				CFVPoints2D(const FVL::CFVPoints2D<T> &copy)    { x = CFVArray<T>(copy.x);  y = CFVArray<T>(copy.y); }

				/**
				 * GETTERS/SETTERS
				 */
				unsigned int size() {
					return x.size();
				}
		};
}

#endif // _H_CUDA_FVPOINTS2D

