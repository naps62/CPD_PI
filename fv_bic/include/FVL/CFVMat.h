/**
 * \file CFVMat.h
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifndef _H_CFVMAT
#define _H_CFVMAT

#include <vector>
using std::vector;

#include "FVL/CFVArray.h"

namespace FVL {

	/**
	 * Generic CUDA-ready array of matrixes class
	 *
	 * Even though it is meant to represent an array of matrixes, it is stored as a vector of arrays, where the vector is 1-dimensional representation of the 2D matrix.
	 *
	 * Instead of accessing elements in the form mat[matrix_index][row][column], elements are accessed as mat[row * width + column][matrix_index]. Method elem() provides an easier way of accessing items in CPU-code, in the form elem(row, column, matrix_index)
	 *
	 * \todo Allow for device-only allocation; alloc single block for whole matrix (allow single mem access instead of 2)
	 */
	template<class T>
		class CFVMat {
			private:
				unsigned int w, h, mat_size;

				vector<CFVArray<T> > mat;	///< Matrix is stored as a single dimension array of vectors
				CFVArray<T*> cuda_mat;		///< Vector of pointers for each CUDA array representing each element of the matrix


			public:
				/************************************************
				 * CONSTRUCTORS
				 ***********************************************/

				/**
				 * Empty constructor
				 */
				CFVMat() { }

				/**
				 * Constructor to create a matrix with given dimensions
				 *
				 * \param w Number of columns
				 * \param h Number of rows
				 * \param size Number of matrixes to allocated
				 */
				CFVMat(unsigned int w, unsigned int h, unsigned int size) { alloc(w, h, size);}

				/**
				 * Default destructor
				 *
				 * Releases all memory allocated for this array
				 */
				~CFVMat() { dealloc(); }

				/************************************************
				 * GETTERS/SETTERS
				 ***********************************************/

				/**
				 * Gives a vector of the matrix values at given coords
				 *
				 * For coords (x, y) gives the vector with elements of each matrix at position (x, y)
				 *
				 * \param x Row of the matrixes to access
				 * \param y Column of the matrixes to access
				 * \return Array of elements contained at (x, y) in each matrix
				 */
				CFVArray<T>& elem(unsigned int x, unsigned int y) { return mat[y * w + x]; }

				// returns single value of matrix at coords (x, y) for index i
				/**
				 * Returns a single element of a matrix at given coordinates
				 *
				 * \param x Row of the element to acess
				 * \param y Column of the element to access
				 * \param i Index of the matrix to acess
				 * \return element of matrix with index i, at coordinates (x, y)
				 */
				T & elem(unsigned int x, unsigned int y, unsigned int i) {
					return (this->elem(x, y))[i];
				}

				/**
				 * Get the vector of each array of the matrix
				 *
				 * \return Vector of arrays of the matrix
				 */
				vector<CFVArray<T> >& getMat() { return mat; }

				/**
				 * Gives number of matrixes stored
				 *
				 * \return Number of matrixes stored
				 */
				unsigned int size() 	const { return mat_size; }

				/**
				 * Gives width (number of columns) of each matrix
				 *
				 * \return Width of each matrix
				 */
				unsigned int width() 	const { return w; }

				/**
				 * Gives height (number of rows) of each matrix
				 *
				 * \return Height of each matrix
				 */
				unsigned int height() 	const { return h; }

				/************************************************
				 * CUDA
				 ***********************************************/
				#ifdef __CUDACC__

				/**
				 * Get the array of pointers for each elem in CUDA memory
				 *
				 * \return Array of pointers for each elem
				 */
				T** cuda_get();

				/**
				 * Allocates space in CUDA memory for all matrixes
				 *
				 * \return Array of pointers for each elem
				 */
				T** cuda_malloc();

				/**
				 * Allocates space in CUDA memory for all matrixes, and copies all data to it
				 *
				 * \param stream CUDA Stream to use (defaults to 0 to use no stream)
				 * \return Array of pointers for each elem
				 */
				T** cuda_mallocAndSave(cudaStream_t stream = 0);

				/**
				 * Releases CUDA memory of all matrixes
				 */
				void cuda_free();

				/**
				 * copies data from the array on host memory to CUDA memory
				 *
				 * \param stream cuda stream to use (defaults to 0 to use no stream)
				 */
				void cuda_save(cudaStream_t stream = 0);

				/**
				 * copies data from the array on CUDA memory back to the host
				 *
				 * \param stream cuda stream to use (defaults to 0 to use no stream)
				 */
				void cuda_load(cudaStream_t stream = 0);

				#endif
			private:

				/************************************************
				 * MEMORY MANAGEMENT
				 ***********************************************/

				/**
				 * Allocate space for matrixes
				 *
				 * \param w Number of columns
				 * \param h Number of rows
				 * \param size Number of matrixes to allocate
				 */
				void alloc(unsigned int w, unsigned int h, unsigned int size);

				/**
				 * Release allocated space for all matrixes
				 */
				void dealloc();
		};
}

#include "FVL/templates/CFVMat.hpp"

#endif // _H_CFVMAT
