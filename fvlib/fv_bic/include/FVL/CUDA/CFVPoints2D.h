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

#include "FVL/CUDA/CFVVect.h"

namespace FVL {

    class CFVPoints2D {
        public:
            /**
             * one array for X coord, another for Y
             */
            CFVVect<double> x, y;

            CFVPoints2D()                                   { }
            CFVPoints2D(const unsigned int size)            { x = CFVVect<double>(size);    y = CFVVect<double>(size); }
            CFVPoints2D(const FVL::CFVPoints2D &copy)    { x = CFVVect<double>(copy.x);  y = CFVVect<double>(copy.y); }

            /**
             * GETTERS/SETTERS
             */
            unsigned int size() {
                return x.size();
            }
    };
}

#endif // _H_CUDA_FVPOINTS2D

