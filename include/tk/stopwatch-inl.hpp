#ifndef ___LIBTK___STOPWATCH_INLINE_HPP
#define ___LIBTK___STOPWATCH_INLINE_HPP

#include <tk/stopwatch.hpp>

namespace tk
{
//////////////////////////////////////////////////	
//
// TIME
//
//////////////////////////////////////////////////
inline
timeval
Time::timernow()
{
	timeval now;
	return timernow( now );
}

inline
timeval&
Time::timernow(timeval& time)
{
	gettimeofday( &time , NULL );
	return time;
}

inline
void
Time::set( const timeval& time )
{
	this->tv_sec = time.tv_sec;
	this->tv_usec = time.tv_usec;
}

inline
void
Time::set( time_t seconds , suseconds_t microseconds )
{
	this->tv_sec = seconds;
	this->tv_usec = microseconds;
}

inline
void
Time::now()
{
	this->set( Time::timernow() );
}

inline
void
Time::reset()
{
	timerclear( this );
}



//
//  GETTERS
//

inline
long long int
Time::microseconds()
const
{
	return this->tv_sec * 1000000 + this->tv_usec;
}

inline
double
Time::miliseconds()
const
{
	return this->tv_sec * 1000 + this->



//////////////////////////////////////////////////	
//
// STOPWATCH
//
//////////////////////////////////////////////////
inline
void
Stopwatch::start()
{
	if ( ! _running )
	{
		




}// namespace tk


#endif//___LIBTK___STOPWATCH_INLINE_HPP
