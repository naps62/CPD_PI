// ------ FVVGausPoint.h ------
// S. CLAIN 2011/11
#ifndef _FVGAUSSPOINT_H
#define _FVGAUSSPOINT_H
//
#include "FVPoint2D.h"
#include "FVPoint3D.h"
#include "FVPoint4D.h"


//
//
//  class for 1D Gauss Point
//
//

class FVGaussPoint1D{
public:
// constructor
    FVGaussPoint1D(){ ; }
    FVPoint2D<double> getPoint(size_t , size_t );
    double getWeight(size_t order, size_t no);
    size_t getNbPoint(size_t order);

};


//
//
//  class for 2D Gauss Point
//
//
class FVGaussPoint2D{
public:
    // constructor
    FVGaussPoint2D() { ; }
    FVPoint3D<double> getPoint(size_t order, size_t no);
    double getWeight(size_t order, size_t no);
    size_t getNbPoint(size_t order);
private:
};

//
//
//  class for 3D Gauss Point
//
//
class FVGaussPoint3D{
public:
    // constructor
    FVGaussPoint3D() { ; }
    FVPoint4D<double> getPoint(size_t order, size_t no);
    double getWeight(size_t order, size_t no);
    size_t getNbPoint(size_t order);
private:
};



#endif // define _FVGAUSSPOINT