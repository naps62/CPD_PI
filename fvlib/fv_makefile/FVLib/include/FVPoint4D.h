#ifndef __FVPOINT4D_H
#define __FVPOINT4D_H

#include <iostream>
using std::ostream;

#include <iomanip>
using std::setw;
using std::ios;

#include <math.h>
#include "FVLib_config.h"
template<class T_>
class FVPoint4D{
public:
T_ x,y,z,t;
// constructor
    FVPoint4D() { x = y = z = t = T_(0); }
    FVPoint4D(T_ a, T_ b=T_(0), T_ c=T_(0),T_  d=T_(0)) { x = a; y = b; z = c; t = d;}
// Copy constructor
    FVPoint4D(const FVPoint4D<T_> &pt) { x = pt.x; y = pt.y; z = pt.z; t = pt.t; }
    void show()
        {   
        std::cout << "( " << std::setprecision(FVPRECISION) << setw(FVCHAMP) << x << " , ";
        std::cout << setw(FVCHAMP) << y << " , " << setw(FVCHAMP) << z<< " , " << setw(FVCHAMP) << t << " )"<<std::endl;
        }
//  Operator += . add with an other FVPoint4D
    FVPoint4D<T_> & operator+=(const FVPoint4D<T_> &p) { x+=p.x; y+=p.y; z+=p.z; t+=p.t;return *this; }
//  Operator -=. substract with an other FVPoint4D
    FVPoint4D<T_> & operator-=(const FVPoint4D<T_> &p) { x-=p.x; y-=p.y; z-=p.z; t-=p.t;return *this; }
//  Operator =. initialize with class T_
    FVPoint4D<T_> & operator=(const T_ &a)  { x = y = z = t = a; return *this; }
//  Operator +=. add with class T_
    FVPoint4D<T_> & operator+=(const T_ &a) { x+=a; y+=a; z+=a; t+=a;return *this; }
//  Operator -=. substract with class T_
    FVPoint4D<T_> & operator-=(const T_ &a) { x-=a; y-=a; z-=a; t-=a;return *this; }
//  Operator *=. multiply with class T_
    FVPoint4D<T_> & operator*=(const T_ &a) { x*=a; y*=a; z*=a; t*=a;return *this; }
//  Operator /=. divide with class T_
    FVPoint4D<T_> & operator/=(const T_ &a) { x/=a; y/=a; z/=a; t/=a;return *this; }


private:
};


///////////////////////////////////////////////////////////////////////////////
//                             ASSOCIATED  FUNCTIONS                         //
///////////////////////////////////////////////////////////////////////////////



template <class T_>
FVPoint4D<T_> operator+ (const FVPoint4D<T_> &a, const FVPoint4D<T_> &b) 
             { return FVPoint4D<T_>(a.x+b.x,a.y+b.y,a.z+b.z,a.t+b.t); }
template <class T_>
FVPoint4D<T_> operator+ (const FVPoint4D<T_> &a, const T_ &x) 
             { return FVPoint4D<T_>(a.x+x,a.y+x,a.z+x,a.t+x); }
template <class T_>
FVPoint4D<T_> operator- (const FVPoint4D<T_> &a) 
             { return FVPoint4D<T_>(-a.x,-a.y,-a.z,-a.t); }
template <class T_>
FVPoint4D<T_> operator- (const FVPoint4D<T_> &a, const FVPoint4D<T_> &b) 
             { return FVPoint4D<T_>(a.x-b.x,a.y-b.y,a.z-b.z,a.t-b.t); }
template <class T_>
FVPoint4D<T_> operator- (const FVPoint4D<T_> &a, const T_ &x) 
            { return FVPoint4D<T_>(a.x-x,a.y-x,a.z-x,a.t-x); }
template <class T_>
FVPoint4D<T_> operator* (const T_ &a, const FVPoint4D<T_> &b) 
            { return FVPoint4D<T_>(a*b.x,a*b.y,a*b.z,a*b.t); }
template <class T_>
FVPoint4D<T_> operator* (const FVPoint4D<T_> &b, const T_ &a) 
            { return FVPoint4D<T_>(a*b.x,a*b.y,a*b.z,a*b.t);}
template <class T_>
T_ operator* (const FVPoint4D<T_> &b, const FVPoint4D<T_> &a) 
            { return (a.x*b.x+a.y*b.y+a.z*b.z+a.t*b.t); }
template <class T_>
FVPoint4D<T_> operator/ (const FVPoint4D<T_> &b, const T_ &a) 
            { return FVPoint4D<T_>(b.x/a,b.y/a,b.z/a,b.t/a);}


template <class T_>
std::ostream & operator<<(std::ostream &s, const FVPoint4D<T_> &a)
{
   s.setf(ios::scientific);
   s << "( " << std::setprecision(FVPRECISION) << setw(FVCHAMP) << a.x << " , ";
   s << setw(FVCHAMP) << a.y << " , " << setw(FVCHAMP) << a.z<< " , " << setw(FVCHAMP) << a.t << " )";
   return s;
}

/* ------ associated functions ----*/


//inline double Det(const FVPoint4D<double> &u, const FVPoint4D<double> &v, const FVPoint4D<double> &w)
//{
//   return (u.x*(v.y*w.z-v.z*w.y)-u.y*(v.x*w.z-v.z*w.x)+u.z*(v.x*w.y-v.y*w.x));
//}

inline double Norm(const FVPoint4D<double> &u)
{
   return sqrt(u.x*u.x+u.y*u.y+u.z*u.z+u.t*u.t);
}

#endif // define _FVPOINT3D