/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVPoints2D.h
 ** CUDA Vector (cuda-optimized storage, struct-of-arrays instead
 **    of array-of-structs
 **
 ** Author: Miguel Palhas, mpalhas@gmail.com
 ** -------------------------------------------------------------------------*/

#ifndef _CUDA_FVVECT
#define _CUDA_FVVECT

#include "MFVLog.h"

namespace CudaFV {

	template<class T>
		class CFVVect {
			private:
				T *arr;
				unsigned int arr_size;

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
				unsigned int size() const;

			private:
				void alloc(unsigned int size);
				void dealloc();

		};

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

	template<class T>
		unsigned int CFVVect<T>::size() const {
			return arr_size;
		}

	template<class T>
		void CFVVect<T>::alloc(unsigned int size) {
			FVLog::logger << "CFVVect alloc(" << size << ")" << endl;
			arr_size = size;
			if (arr_size > 0) {
				arr = new T[arr_size];
				}
		}

	template<class T>
		void CFVVect<T>::dealloc() {
			FVLog::logger << "CFVVect dealloc()" << endl;
			if (arr != NULL) {
				delete arr;
			}
			arr = NULL;
			arr_size = 0;
		}
}
#endif // _CUDA_FVVECT
