#ifndef __FVPOINT3D_H
#define __FVPOINT3D_H

#include <iostream>
using std::ostream;

#include <iomanip>
using std::setw;
using std::ios;

#include <math.h>
#include "FVLib_config.h"
template<class T_>
class FVPoint3D{
public:
T_ x,y,z;
// constructor
    FVPoint3D() { x = y = z = T_(0); }
    FVPoint3D(T_ a, T_ b=T_(0), T_ c=T_(0)) { x = a; y = b; z = c; }
// Copy constructor
    FVPoint3D(const FVPoint3D<T_> &pt) { x = pt.x; y = pt.y; z = pt.z; }
    void show()
        {   
        std::cout << "( " << std::setprecision(FVPRECISION) << setw(FVCHAMP) << x << " , ";
        std::cout << setw(FVCHAMP) << y << " , " << setw(FVCHAMP) << z<<  " )"<<std::endl;
        }   
   
//  Operator += . add with an other FVPoint3D
    FVPoint3D<T_> & operator+=(const FVPoint3D<T_> &p) { x+=p.x; y+=p.y; z+=p.z; return *this; }
//  Operator -=. substract with an other FVPoint3D
    FVPoint3D<T_> & operator-=(const FVPoint3D<T_> &p) { x-=p.x; y-=p.y; z-=p.z; return *this; }
//  Operator =. initialize with class T_
    FVPoint3D<T_> & operator=(const T_ &a)  { x = y = z = a; return *this; }
//  Operator +=. add with class T_
    FVPoint3D<T_> & operator+=(const T_ &a) { x+=a; y+=a; z+=a; return *this; }
//  Operator -=. substract with class T_
    FVPoint3D<T_> & operator-=(const T_ &a) { x-=a; y-=a; z-=a; return *this; }
//  Operator *=. multiply with class T_
    FVPoint3D<T_> & operator*=(const T_ &a) { x*=a; y*=a; z*=a; return *this; }
//  Operator /=. divide with class T_
    FVPoint3D<T_> & operator/=(const T_ &a) { x/=a; y/=a; z/=a; return *this; }


private:
};


///////////////////////////////////////////////////////////////////////////////
//                             ASSOCIATED  FUNCTIONS                         //
///////////////////////////////////////////////////////////////////////////////



template <class T_>
FVPoint3D<T_> operator+ (const FVPoint3D<T_> &a, const FVPoint3D<T_> &b) 
             { return FVPoint3D<T_>(a.x+b.x,a.y+b.y,a.z+b.z); }
template <class T_>
FVPoint3D<T_> operator+ (const FVPoint3D<T_> &a, const T_ &x) 
             { return FVPoint3D<T_>(a.x+x,a.y+x,a.z+x); }
template <class T_>
FVPoint3D<T_> operator- (const FVPoint3D<T_> &a) 
             { return FVPoint3D<T_>(-a.x,-a.y,-a.z); }
template <class T_>
FVPoint3D<T_> operator- (const FVPoint3D<T_> &a, const FVPoint3D<T_> &b) 
             { return FVPoint3D<T_>(a.x-b.x,a.y-b.y,a.z-b.z); }
template <class T_>
FVPoint3D<T_> operator- (const FVPoint3D<T_> &a, const T_ &x) 
            { return FVPoint3D<T_>(a.x-x,a.y-x,a.z-x); }
template <class T_>
FVPoint3D<T_> operator* (const T_ &a, const FVPoint3D<T_> &b) 
            { return FVPoint3D<T_>(a*b.x,a*b.y,a*b.z); }
template <class T_>
FVPoint3D<T_> operator* (const FVPoint3D<T_> &b, const T_ &a) 
            { return FVPoint3D<T_>(a*b.x,a*b.y,a*b.z);}
template <class T_>
T_ operator* (const FVPoint3D<T_> &b, const FVPoint3D<T_> &a) 
            { return (a.x*b.x+a.y*b.y+a.z*b.z); }
template <class T_>
FVPoint3D<T_> operator/ (const FVPoint3D<T_> &b, const T_ &a) 
            { return FVPoint3D<T_>(b.x/a,b.y/a,b.z/a);}


template <class T_>
std::ostream & operator<<(std::ostream &s, const FVPoint3D<T_> &a)
{
   s.setf(ios::scientific);
   s << "( " << std::setprecision(FVPRECISION) << setw(FVCHAMP) << a.x << " , ";
   s << setw(FVCHAMP) << a.y << " , " << setw(FVCHAMP) << a.z << " )";
   return s;
}

/* ------ associated functions ----*/


inline FVPoint3D<double> CrossProduct(const FVPoint3D<double> &u, const FVPoint3D<double> &v)
{
   return FVPoint3D<double>(u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x);
}

inline double Det(const FVPoint3D<double> &u, const FVPoint3D<double> &v, const FVPoint3D<double> &w)
{
   return (u.x*(v.y*w.z-v.z*w.y)-u.y*(v.x*w.z-v.z*w.x)+u.z*(v.x*w.y-v.y*w.x));
}

inline double Norm(const FVPoint3D<double> &u)
{
   return sqrt(u.x*u.x+u.y*u.y+u.z*u.z);
}

#endif // define _FVPOINT3D