/**
 * \file FVArray.hpp
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 */

#ifdef _H_CFVARRAY

#ifndef _HPP_FVARRAY
#define _HPP_FVARRAY

namespace FVL {

	/************************************************
	 * CONSTRUCTORS
	 ***********************************************/

	template<class T>
		FVArray<T>::FVArray(const unsigned int size) {
			alloc(size);
		}

	template<class T>
		FVArray<T>::FVArray(const FVArray<T> &copy) {
			alloc(copy.size());
			for(unsigned int i = 0; i < arr_size; ++i) {
				arr[i] = copy[i];
			}
		}

	template<class T>
		FVArray<T>::~FVArray() {
			dealloc();
		}

	/************************************************
	 * OPERATORS
	 ***********************************************/

	template<class T>
		T & FVArray<T>::operator [] (int index) {
			return arr[index];
		}

	template<class T>
		const T & FVArray<T>::operator [] (int index) const {
			return arr[index];
		}

	template<class T>
		const FVArray<T> & FVArray<T>::operator = (const FVArray<T> & copy) {
			dealloc();
			alloc(copy.size());
			for(unsigned int i = 0; i < arr_size; ++i) {
				arr[i] = copy[i];
			}
			return *this;
		}

	/************************************************
	 * GETTERS/SETTERS
	 ***********************************************/

	template<class T>
		T* FVArray<T>::getArray() {
			return arr;
		}

	template<class T>
		unsigned int FVArray<T>::size() const {
			return arr_size;
		}

	/************************************************
	 * MEMORY MANAGEMENT
	 ***********************************************/

	template<class T>
		void FVArray<T>::alloc(unsigned int size) {
			arr_size = size;
			if (arr_size > 0) {
				arr = new T[arr_size];
			}
		}

	template<class T>
		void FVArray<T>::dealloc() {
			// deallocate only if previously allocated
			if (arr != NULL) {
				delete arr;
			}
			arr = NULL;
			arr_size = 0;
		}
}

#endif // _HPP_FVARRAY
#endif // _H_FVARRAY

