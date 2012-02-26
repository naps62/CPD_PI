/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVMat.h
 ** CUDA Matrixes (cuda-optimized storage, struct-of-arrays instead
 **    of array-of-structs
 ** 
 ** Intended to store an array of matrixes (but as s-o-a instead of a-o-s. e.g., store a matrix for each cell of a mesh
 **
 ** Author: Miguel Palhas, mpalhas@gmail.com
 ** TODO: allow for device-only allocation
 ** -------------------------------------------------------------------------*/

#ifndef _H_CFVMAT
#define _H_CFVMAT

#include <vector>
using std::vector;

#include "FVL/CFVVect.h"

namespace FVL {

	template<class T>
	class CFVMat {
		private:
			unsigned int w, h, mat_size;

			/**
			 * Matrix is stored as a single dimension array of vectors
			 */
			vector<CFVVect<T> > mat;
			CFVVect<T*> cuda_mat;

		public:
			/**
			 * CONSTRUCTORS
			 */
			CFVMat(unsigned int w, unsigned int h, unsigned int size) { alloc(w, h, size);}
			~CFVMat() { dealloc(); }

			/**
			 * GETTERS/SETTERS
			 */

			// returns a vector of the matrix values at coords (x,y)
			CFVVect<T>& elem(unsigned int x, unsigned int y) { return mat[y * w + x]; }

			// returns single value of matrix at coords (x, y) for index i
			T & elem(unsigned int x, unsigned int y, unsigned int i) {
				return (this->elem(x, y))[i];
			}

			vector<CFVVect<T> >& getMat() { return mat; }
			unsigned int size() 	const { return mat_size; }
			unsigned int width() 	const { return w; }
			unsigned int height() 	const { return h; }

			/**
			 * CUDA
			 *
			 * Each elem is stored as an individual vector
			 * an additional T** array of size w*y is stored to hold references to each elem
			 */
	
			// get the array of pointers to each elem
			T** cuda_getMat();
			// alloc cuda mem
			T** cuda_malloc();
			// alloc and copy data
			T** cuda_mallocAndSave();

			// free cuda mem
			void cuda_free();
			
			// save to device (memcpyHostToDevice)
			void cuda_save();
			void cuda_saveAsync(cudaStream_t &stream);

			// get from device (memcpyDeviceToHost)
			void cuda_get();
		
		private:
			/**
			 * ALLOC/DEALLOC
			 */
			void alloc(unsigned int w, unsigned int h, unsigned int size);
			void dealloc();
	};
}

#include "FVL/templates/CFVMat.hpp"

#endif // _H_CFVMAT
