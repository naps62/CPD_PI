/* ---------------------------------------------------------------------------
 ** Finite Volume Library 
 **
 ** CFVPoints2D.h
 ** CUDA Vector of 2D Points (cuda-optimized storage, struct-of-arrays instead
 **    of array-of-structs
 **
 ** Author: Miguel Palhas, mpalhas@gmail.com
 ** -------------------------------------------------------------------------*/

#ifndef _H_CUDA_FVPOINTS2D
#define _H_CUDA_FVPOINTS2D

#include "CUDA/CFVVect.h"

namespace CudaFV {

    class CFVPoints2D {
        public:
            /**
             * one array for X coord, another for Y
             */
            CFVVect<fv_float> x, y;

            CFVPoints2D()                                   { }
            CFVPoints2D(const unsigned int size)            { x = CFVVect<fv_float>(size);    y = CFVVect<fv_float>(size); }
            CFVPoints2D(const CudaFV::CFVPoints2D &copy)    { x = CFVVect<fv_float>(copy.x);  y = CFVVect<fv_float>(copy.y); }

            /**
             * GETTERS/SETTERS
             */
            unsigned int size() {
                return x.size();
            }
    };
}

#endif // _H_CUDA_FVPOINTS2D
