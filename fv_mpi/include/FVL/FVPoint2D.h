/**
 * \file FVLib.h
 *
 * \brief Global header file for FVL.
 *
 * Use this to include all library headers
 *
 * \author Miguel Palhas
 * \date 13-02-2012
 * \todo operators need testing (just to be sure)
 */

#ifndef _H_FVPOINT2D
#define _H_FVPOINT2D

#include <cmath>
#include <iostream>
#include <iomanip>

#include "FVL/FVGlobal.h"

namespace FVL {

	/**
	 * 2 Dimensional point template class
	 *
	 * Manages a single point in a 2D space
	 */
	template<class T>
	class FVPoint2D {

		public:
			T x; ///< X coord of the point
			T y; ///< Y coord of the point


			/************************************************
			 * CONSTRUCTORS
			 ***********************************************/

			/**
			 * Default constructor. Initializes point to (0, 0)
			 */
			FVPoint2D() 						{x = y = T(0); }

			/**
			 * Constructor from two parameter values, for X and Y respectively
			 */
			FVPoint2D(T new_x, T new_y) 		{x = new_x; y = new_y;}

			/**
			 * Copy constructor
			 */
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

			/************************************************
			 * MATH
			 ***********************************************/
			/**
			 * Gives the determinant between this point and a given point
			 *
			 * Being A the main point and B the point given as argument, the determinant is given by:
			 * \f$ \sqrt{(x_2-x_1)^2+(y_2-y_1)^2} \f$
			 *
			 * \param b The point to use
			 */
			inline T determinant(const FVPoint2D<T> &b) { return (this->x * b.y - this->y * b.x);}

			/**
			 * Gives the norm of the point
			 *
			 * The norm is given by:
			 * \f$ Norm = \sqrt{x*x + y*y} \f$
			 */
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
