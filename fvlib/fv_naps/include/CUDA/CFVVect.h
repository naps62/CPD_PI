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

namespace CudaFV {

	template<class T>
		class CFVVect {
			private:
				/**
				 * size of each array
				 */
				unsigned int arr_size;

				/**
				 * one array of doubles for each dimension
				 */
				T *arr;

			public:
				/**
				 * CONSTRUCTORS
				 */
				CFVVect();
				CFVVect(const unsigned int size);
				CFVVect(const CFVVect<T> &copy);

				~CFVVect() { dealloc(); }

				/**
				 * OPERATORS
				 */
				T & operator[](const unsigned int index);

				/**
				 * GETTERS/SETTERS
				 */
				unsigned int size();

				/**
				 * ALLOC/DELETE
				 */
				void alloc(const unsigned int new_size);

				void dealloc(); 

			private:
				void init();

		};

	template<class T>
		CFVVect<T>::CFVVect() {
			init();
		}

	template<class T>
		CFVVect<T>::CFVVect(const unsigned int size) {
			init();
			alloc(size);
		}

	template<class T>
		T & CFVVect<T>::operator[](const unsigned int index) {
			return arr[index];
		}
	template<class T>
		CFVVect<T>::CFVVect(const CFVVect<T> &copy) {
			for(unsigned int i = 0; i < arr_size; ++i) {
				(*this)[i] = copy[i];
			}
		}


	template<class T>
		unsigned int CFVVect<T>::size() {
			return arr_size;
		}

	template<class T>
		void CFVVect<T>::alloc(const unsigned int new_size) {
			if (size() != 0)
				dealloc();
			arr = new T[new_size];
			this->arr_size = new_size;
		}

	template<class T>
		void CFVVect<T>::dealloc() {
			if (arr_size == 0)
				return;
			delete arr;
			arr = NULL;
			arr_size = 0;
		}

	template<class T>
		void CFVVect<T>::init() {
			arr_size = 0;
			arr = NULL;
		}
}
#endif // _CUDA_FVVECT
