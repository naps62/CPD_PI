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



//
//  STATIC
//

inline
timeval Time::timernow()
{
	timeval now;
	return timernow( now );
}

inline
timeval& Time::timernow(timeval& time)
{
	gettimeofday( &time , NULL );
	return time;
}



//
//  CONSTRUCTORS
//

inline
Time::Time() { this->now(); }

inline
Time::Time(const Time& original) { this->set( original ); }



//
//  GETTERS
//

inline
long long int
Time::microseconds()
const
{ return this->tv_sec * 1000000 + this->tv_usec; }

inline
double
Time::miliseconds()
const
{ return this->tv_sec * 1e3 + this->tv_usec * 1e-3; }

inline
double
Time::seconds()
const
{ return this->tv_sec + tv_usec * 1e-6; }



//
//	SETTERS
//

inline
void Time::set( const timeval& time )
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
{ this->set( Time::timernow() ); }

inline
void
Time::reset()
{ timerclear( this ); }



//
//  OPERATORS
//

inline
Time& Time::operator=(const Time& time)
{
	this->set(time);
	return (*this);
}

inline
Time&
Time::operator+=(const Time& time)
{
	timeval s;
	timeradd(this,&time,&s);
	this->set(s);
	return *this;
}

inline
Time&
Time::operator-=(const Time& time)
{
	timeval s;
	timersub(this,&time,&s);
	this->set(s);
	return *this;
}

inline
const Time
Time::operator+(const Time &time)
{
	Time t = *this;
	t += time;
	return t;
}

inline
const Time
Time::operator-(const Time &time)
{
	Time t = *this;
	t -= time;
	return t;
}



//
//  FRIENDS
//

inline
const Time operator-( const Time& t1 , const Time& t2 )
{
	Time t = t1;
	t -= t2;
	return t;
}
	


inline
std::ostream& 
operator<<(std::ostream& out, const Time& time)
{
	out
		<<	'['
		<<	time.get_seconds()
		<<	'+'
		<<	time.get_microseconds()
		<<	']';
	return out;
}





//////////////////////////////////////////////////	
//
// STOPWATCH
//
//////////////////////////////////////////////////
inline
Stopwatch::Stopwatch()
: _running(false)
{
	_total.reset();
	_control.reset();
	this->start();
	this->stop();
	_control = _total;
	_total.reset();
}

inline
void Stopwatch::start()
{
	if ( ! _running )
	{
		_last.reset();
		_running = true;
		_begin.now();
	}
}

inline
void
Stopwatch::stop()
{
	_end.now();
	if ( _running )
	{
		_running = false;
		_last = ( _end - _begin ) - _control;
		_total += _last;
	}
}

inline
void
Stopwatch::reset()
{
	_last.reset();
	if ( ! _running )
		_total.reset();
}

//
//  GETTERS
//
inline
Time Stopwatch::total() { return _total; }

inline
Time Stopwatch::last() { return _last; }



}// namespace tk


#endif//___LIBTK___STOPWATCH_INLINE_HPP
