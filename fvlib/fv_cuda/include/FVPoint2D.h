#ifndef __FVPOINT2D_H
#define __FVPOINT2D_H

#include <iostream>
using std::ostream;

#include <iomanip>
using std::setw;
using std::ios;

#include <math.h>
#include "FVLib_config.h"
template<class T_>
class FVPoint2D{
public:
T_ x,y;
// constructor
    FVPoint2D() { x = y =  T_(0); }
    FVPoint2D(T_ a, T_ b=T_(0)) { x = a; y = b; }
// Copy constructor
    FVPoint2D(const FVPoint2D<T_> &pt) { x = pt.x; y = pt.y;  }
    void show()
        {   
        std::cout << "( " << std::setprecision(FVPRECISION) << setw(FVCHAMP) << x << " , ";
        std::cout << setw(FVCHAMP) << y  << " )"<<std::endl;
        }    
   
//  Operator += . add with an other FVPoint2D
    FVPoint2D<T_> & operator+=(const FVPoint2D<T_> &p) { x+=p.x; y+=p.y; return *this; }
//  Operator -=. substract with an other FVPoint2D
    FVPoint2D<T_> & operator-=(const FVPoint2D<T_> &p) { x-=p.x; y-=p.y;  return *this; }
//  Operator =. initialize with class T_
    FVPoint2D<T_> & operator=(const T_ &a)  { x = y = a; return *this; }
//  Operator +=. add with class T_
    FVPoint2D<T_> & operator+=(const T_ &a) { x+=a; y+=a;  return *this; }
//  Operator -=. substract with class T_
    FVPoint2D<T_> & operator-=(const T_ &a) { x-=a; y-=a;  return *this; }
//  Operator *=. multiply with class T_
    FVPoint2D<T_> & operator*=(const T_ &a) { x*=a; y*=a;  return *this; }
//  Operator /=. divide with class T_
    FVPoint2D<T_> & operator/=(const T_ &a) { x/=a; y/=a;  return *this; }

     
        
private:
};


///////////////////////////////////////////////////////////////////////////////
//                             ASSOCIATED  FUNCTIONS                         //
///////////////////////////////////////////////////////////////////////////////

template <class T_>
inline T_ Det(const FVPoint2D<T_> &u,const FVPoint2D<T_> &v)
        {return (u.x*v.y-v.x*u.y);}

template <class T_>
inline T_ Norm(const FVPoint2D<T_> &u)
        {return sqrt(u.x*u.x+u.y*u.y);}  

template <class T_>
FVPoint2D<T_> operator+ (const FVPoint2D<T_> &a, const FVPoint2D<T_> &b) 
             { return FVPoint2D<T_>(a.x+b.x,a.y+b.y); }
template <class T_>
FVPoint2D<T_> operator+ (const FVPoint2D<T_> &a, const T_ &x) 
             { return FVPoint2D<T_>(a.x+x,a.y+x); }
template <class T_>
FVPoint2D<T_> operator- (const FVPoint2D<T_> &a) 
             { return FVPoint2D<T_>(-a.x,-a.y); }
template <class T_>
FVPoint2D<T_> operator- (const FVPoint2D<T_> &a, const FVPoint2D<T_> &b) 
             { return FVPoint2D<T_>(a.x-b.x,a.y-b.y); }
template <class T_>
FVPoint2D<T_> operator- (const FVPoint2D<T_> &a, const T_ &x) 
            { return FVPoint2D<T_>(a.x-x,a.y-x); }
template <class T_>
FVPoint2D<T_> operator* (const T_ &x, const FVPoint2D<T_> &b) 
            { return FVPoint2D<T_>(x*b.x,x*b.y); }
template <class T_>
FVPoint2D<T_> operator* (const FVPoint2D<T_> &b, const T_ &x) 
            { return FVPoint2D<T_>(x*b.x,x*b.y);}
template <class T_>
T_ operator* (const FVPoint2D<T_> &b, const FVPoint2D<T_> &a) 
            { return (a.x*b.x+a.y*b.y); }
template <class T_>
FVPoint2D<T_> operator/ (const FVPoint2D<T_> &b, const T_ &x) 
            { return FVPoint2D<T_>(b.x/x,b.y/x);}




template <class T_>
std::ostream & operator<<(std::ostream &s, const FVPoint2D<T_> &a)
{
   s.setf(ios::scientific);
   s << "( " << std::setprecision(FVPRECISION) << setw(FVCHAMP) << a.x << " , ";
   s << setw(FVCHAMP) << a.y << " )";
   return s;
}



#endif // define _FVPOINT2D
