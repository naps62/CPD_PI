/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVVect.hpp
 ** Template declaration for CFVVect.h
 **
 ** Author: Miguel Palhas, mpalhas@gmail.com
 ** -------------------------------------------------------------------------*/

#ifdef _H_CUDA_FVVECT

#include <cuda_runtime_api.h>
#include <cuda.h>
//#include <cutil.h>

namespace FVL {

	/**
	 * CONSTRUCTORS
	 */
	template<class T>
		CFVVect<T>::CFVVect() {
			arr = NULL;
			alloc(0);
		}

	template<class T>
		CFVVect<T>::CFVVect(const unsigned int size) {
			alloc(size);
		}

	template<class T>
		CFVVect<T>::CFVVect(const CFVVect<T> &copy) {
			alloc(copy.size());
			for(unsigned int i = 0; i < arr_size; ++i) {
				arr[i] = copy[i];
			}
		}

	template<class T>
		CFVVect<T>::~CFVVect() {
			dealloc();
		}

	/**
	 * OPERATORS
	 */
	template<class T>
		T & CFVVect<T>::operator [] (int index) {
			return arr[index];
		}

	template<class T>
		const T & CFVVect<T>::operator [] (int index) const {
			return arr[index];
		}

	template<class T>
		const CFVVect<T> & CFVVect<T>::operator = (const CFVVect<T> & copy) {
			dealloc();
			alloc(copy.size());
			for(unsigned int i = 0; i < arr_size; ++i) {
				arr[i] = copy[i];
			}
			return *this;
		}

	/**
	 * GETTERS/SETTERS
	 */
	template<class T>
		T* CFVVect<T>::getArray() {
			return arr;
		}

	template<class T>
		unsigned int CFVVect<T>::size() const {
			return arr_size;
		}

	/**
	 * CUDA
	 */
	template<class T>
		T* CFVVect<T>::cuda_getArray() {
			return cuda_arr;
		}

	template<class T>
		T* CFVVect<T>::cuda_mallocAndSave() {
			this->cuda_malloc();
			this->cuda_save();
			return this->cuda_getArray();
		}

	template<class T>
		void CFVVect<T>::cuda_malloc() {
			cudaMalloc((void **)&cuda_arr, sizeof(T) * arr_size);
		}

	template<class T>
		void CFVVect<T>::cuda_free() {
			cudaFree(cuda_arr);
		}

	template<class T>
		void CFVVect<T>::cuda_save() {
			cudaMemcpy(cuda_arr, arr, sizeof(T) * arr_size, cudaMemcpyHostToDevice);
		}

	template<class T>
		void CFVVect<T>::cuda_saveAsync(cudaStream_t &stream) {
			cudaMemcpyAsync(cuda_arr, arr, sizeof(T) * arr_size, cudaMemcpyHostToDevice, stream);
		}

	template<class T>
		void CFVVect<T>::cuda_get() {
			cudaMemcpy(arr, cuda_arr, sizeof(T) * arr_size, cudaMemcpyDeviceToHost);
		}

	/**
	 * ALLOC/DELETE
	 */
	template<class T>
		void CFVVect<T>::alloc(unsigned int size) {
			arr_size = size;
			if (arr_size > 0) {
				arr = new T[arr_size];
			}
		}

	template<class T>
		void CFVVect<T>::dealloc() {
			if (arr != NULL) {
				delete arr;
			}
			arr = NULL;
			arr_size = 0;
		}
}

#endif

