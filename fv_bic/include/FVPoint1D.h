#ifndef __FVPOINT1D_H
#define __FVPOINT1D_H

#include <iostream>
using std::ostream;

#include <iomanip>
using std::setw;
using std::ios;

#include <math.h>
#include "FVLib_config.h"
template<class T_>
class FVPoint1D{
public:
T_ x;
// constructor
    FVPoint1D() { x =  T_(0); }
    FVPoint1D(T_ a) { x = a; }
// Copy constructor
    FVPoint1D(const FVPoint1D<T_> &pt) { x = pt.x;  }
    void show()
        {   
        std::cout << "( " << std::setprecision(FVPRECISION) << setw(FVCHAMP) << x << " )"<<std::endl;
        }    
   
//  Operator += . add with an other FVPoint1D
    FVPoint1D<T_> & operator+=(const FVPoint1D<T_> &p) { x+=p.x; return *this; }
//  Operator -=. substract with an other FVPoint1D
    FVPoint1D<T_> & operator-=(const FVPoint1D<T_> &p) { x-=p.x;   return *this; }
//  Operator =. initialize with class T_
    FVPoint1D<T_> & operator=(const T_ &a)  { x = a; return *this; }
//  Operator +=. add with class T_
    FVPoint1D<T_> & operator+=(const T_ &a) { x+=a;  return *this; }
//  Operator -=. substract with class T_
    FVPoint1D<T_> & operator-=(const T_ &a) { x-=a;  return *this; }
//  Operator *=. multiply with class T_
    FVPoint1D<T_> & operator*=(const T_ &a) { x*=a;  return *this; }
//  Operator /=. divide with class T_
    FVPoint1D<T_> & operator/=(const T_ &a) { x/=a;  return *this; }

     
        
private:
};


///////////////////////////////////////////////////////////////////////////////
//                             ASSOCIATED  FUNCTIONS                         //
///////////////////////////////////////////////////////////////////////////////


inline double Norm(const FVPoint1D<double> &u)
        {if(u.x<0) return (-u.x); else return(u.x); }  

template <class T_>
FVPoint1D<T_> operator+ (const FVPoint1D<T_> &a, const FVPoint1D<T_> &b) 
             { return FVPoint1D<T_>(a.x+b.x); }
template <class T_>
FVPoint1D<T_> operator+ (const FVPoint1D<T_> &a, const T_ &x) 
             { return FVPoint1D<T_>(a.x+x); }
template <class T_>
FVPoint1D<T_> operator- (const FVPoint1D<T_> &a) 
             { return FVPoint1D<T_>(-a.x); }
template <class T_>
FVPoint1D<T_> operator- (const FVPoint1D<T_> &a, const FVPoint1D<T_> &b) 
             { return FVPoint1D<T_>(a.x-b.x); }
template <class T_>
FVPoint1D<T_> operator- (const FVPoint1D<T_> &a, const T_ &x) 
            { return FVPoint1D<T_>(a.x-x); }
template <class T_>
FVPoint1D<T_> operator* (const T_ &x, const FVPoint1D<T_> &b) 
            { return FVPoint1D<T_>(x*b.x); }
template <class T_>
FVPoint1D<T_> operator* (const FVPoint1D<T_> &b, const T_ &x) 
            { return FVPoint1D<T_>(x*b.x);}
template <class T_>
T_ operator* (const FVPoint1D<T_> &b, const FVPoint1D<T_> &a) 
            { return (a.x*b.x); }
template <class T_>
FVPoint1D<T_> operator/ (const FVPoint1D<T_> &b, const T_ &x) 
            { return FVPoint1D<T_>(b.x/x);}




template <class T_>
std::ostream & operator<<(std::ostream &s, const FVPoint1D<T_> &a)
{
   s.setf(ios::scientific);
   s << "( " << std::setprecision(FVPRECISION) << setw(FVCHAMP) << a.x << " )";
   return s;
}



#endif // define _FVPOINT1D 

