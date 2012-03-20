/* ---------------------------------------------------------------------------
** Finite Volume Library 
**
** Filename: FVPoint2D.h
** 2D Point
**
** Author:		Miguel Palhas, mpalhas@gmail.com
** Created:		13-02-2012
** Last Tested:	---
** Known Issues: operators need testing (just to be sure)
** -------------------------------------------------------------------------*/

#ifndef _H_FVPOINT2D
#define _H_FVPOINT2D

#include <cmath>
#include <iostream>
#include <iomanip>

#include "FVL/FVGlobal.h"

namespace FVL {
	template<class T>
	class FVPoint2D {

		public:
			T x, y;


			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/
			FVPoint2D() 						{x = y = T(0); }
			FVPoint2D(T new_x, T new_y) 		{x = new_x; y = new_y;}
			FVPoint2D(const FVPoint<T> &copy)	{x = copy.x; y = copy.y;}

			/************************************************
			 * OPERATORS
			 ***********************************************/
			FVPoint2D<T> & operator +=	(const FVPoint2D<T> &p)	{ x += p.x; y += p.y; return *this;}
			FVPoint2D<T> & operator -=	(const FVPoint2D<T> &p)	{ x -= p.x; y -= p.y; return *this;}

			// operators for T parameter
			FVPoint2D<T> & operator =	(const T &a) { x = a;	y = a;	return *this;}
			FVPoint2D<T> & operator +=	(const T &a) { x += a;	y += a;	return *this;}
			FVPoint2D<T> & operator -=	(const T &a) { x -= a;	y -= a;	return *this;}
			FVPoint2D<T> & operator *=	(const T &a) { x *= a;	y *= a;	return *this;}
			FVPoint2D<T> & operator /=	(const T &a) { x /= a;	y /= a;	return *this;}


			// Operatros with FVPoint2D & FVpoint2D params
			/*friend FVPoint2D<T> operator + (const friend FVPoint2D<T> &a, const friend FVPoint2D<T> &b);
			friend FVPoint2D<T> operator - (const friend FVPoint2D<T> &a, const friend FVPoint2D<T> &b);
			friend FVPoint2D<T> operator * (const friend FVPoint2D<T> &a, const friend FVPoint2D<T> &b);
			friend FVPoint2D<T> operator / (const friend FVPoint2D<T> &a, const friend FVPoint2D<T> &b);

			// operators with FVPoint2D & T params
			friend FVPoint2D<T> operator + (const friend FVPoint2D<T> &a, const T &x);
			friend FVPoint2D<T> operator - (const friend FVPoint2D<T> &a, const T &x);
			friend FVPoint2D<T> operator * (const friend FVPoint2D<T> &a, const T &x);
			friend FVPoint2D<T> operator * (const T &x, const friend FVpoint2D<T> &a);
			friend FVPoint2D<T> operator / (const friend FVPoint2D<T> &a, const T &x);

			friend FVPoint2D<T> operator - (const FVPoint2D<T> &a);

			friend std::ostream & operator << (std::ostream &s, const FVPoint2D<T> &p);*/

			/************************************************
			 * MATH
			 ***********************************************/
			// calcs the determinant between this point and b
			inline T determinant(const FVPoint2D<T> &b) { return (this->x * b.y - this->y * b.x);}

			// norm of the point
			inline T norm() { return sqrt(x*x + y*y);}
	};

	/************************************************
	 * GLOBAL OPERATORS
	 ***********************************************/

	// Operators with FVPoint2D & FVPoint2D params
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator + (const FVPoint2D<T> &a, const FVPoint2D<T> &b) {return FVPoint2D<T>(a.x+b.x, a.y+b.y);}
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator - (const FVPoint2D<T> &a, const FVPoint2D<T> &b) {return FVPoint2D<T>(a.x-b.x, a.y-b.y);}
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator * (const FVPoint2D<T> &a, const FVPoint2D<T> &b) {return FVPoint2D<T>(a.x*b.x, a.y*b.y);}
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator / (const FVPoint2D<T> &a, const FVPoint2D<T> &b) {return FVPoint2D<T>(a.x/b.x, a.y/b.y);}

	// operators with FVPoint2D & T params
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator + (const FVPoint2D<T> &a, const T &x) {return FVPoint2D<T>(a.x+x, a.y+x);}
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator - (const FVPoint2D<T> &a, const T &x) {return FVPoint2D<T>(a.x-x, a.y-x);}
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator * (const FVPoint2D<T> &a, const T &x) {return FVPoint2D<T>(a.x*x, a.y*x);}
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator * (const T &x, const FVpoint2D<T> &a) {return FVPoint2D<T>(a.x*x, a.y*x);}
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator / (const FVPoint2D<T> &a, const T &x) {return FVPoint2D<T>(a.x/x, a.y/x);}

	// operators with T params
	/// \relates FVPoint2D
	template<class T> FVPoint2D<T> operator - (const FVPoint2D<T> &a) {return FVPoint2D<T>(-a.x, -a.y);}

	// stream operators
	/// \relates FVPoint2D
	template<class T> std::ostream & operator << (std::ostream &s, const FVPoint2D<T> &p) {
		s.setf(std::ios::scientific);
		s	<< std::setprecision(FV_PRECISION) << setw(FV_CHAMP)
			<< "(" << p.x << ", " << p.y << ")";
		return s;
	}
	/**
	 * @}
	 */

}


#endif // _H_FVPOINT2D
