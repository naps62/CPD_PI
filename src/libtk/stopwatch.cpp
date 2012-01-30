#include "stopwatch.hpp"

#include <sys/time.h>

#define	NANOS_PER_SEC	1000000000

//	BEGIN CLASS
timespec Stopwatch::add( const timespec& time1 , const timespec& time2 )
{
	timespec sum;
	long dnsec = time1.tv_nsec + time2.tv_nsec;
	if ( dnsec > NANOS_PER_SEC )
	{
		sum.tv_sec = time1.tv_sec + time2.tv_sec + 1;
		sum.tv_nsec = dnsec - NANOS_PER_SEC;
	}
	else
	{
		sum.tv_sec = time1.tv_sec + time2.tv_sec;
		sum.tv_nsec = dnsec;
	}
	return sum;
}

timespec Stopwatch::diff( const timespec& start , const timespec& end )
{
	timespec diff;
	long dnsec = end.tv_nsec - start.tv_nsec;
	if ( dnsec < 0 )
	{
		diff.tv_sec = end.tv_sec - start.tv_sec - 1;
		diff.tv_nsec = NANOS_PER_SEC + dnsec;
	}
	else
	{
		diff.tv_sec = end.tv_sec - start.tv_sec;
		diff.tv_nsec = dnsec;
	}
	return diff;
}

void Stopwatch::reset( timespec& time )
{
	time.tv_sec = 0;
	time.tv_nsec = 0;
}

//	BEGIN INSTANCE
Stopwatch::Stopwatch() : _total(0) , _running(0)
{}

void Stopwatch::start()
{
	if ( ! _running )
	{
		Stopwatch::reset( _partial );
		_running = ( ! _running );
		clock_gettime( CLOCK_PROCESS_CPUTIME_ID , &_begin );
	}
}

void Stopwatch::stop()
{
	clock_gettime( CLOCK_PROCESS_CPUTIME_ID , &_end );
	if ( _running )
	{
		_running = ( ! _running );
		_partial = Stopwatch::diff( _begin , _end );
		_total = Stopwatch::add( _total , _partial );
	}
}

void Stopwatch::reset()
{
	Stopwatch::reset( _partial );
	if ( ! _running )
		Stopwatch::reset( _total );
}

void Stopwatch::finish()
{
	this->stop();
	this->reset();
}

//	getters
timespec Stopwatch::total()
{
	return _total;
}

double Stopwatch::total_s()
{
	return this->total_ns() / 1e9;

time_t Stopwatch::total_s()
{
	return _total.tv_sec;
}

long long int Stopwatch::total_ns()
{
	return _total.tv_sec * NANOS_PER_SEC + _total.tv_nsec;
}

double Stopwatch::total_us()
{
	return this->total_ns() / 1e3;
}

long long int Stopwatch::total_us()
{
	return this->total_ns() / 1000;
}

timespec Stopwatch::partial()
{
	return _partial;
}
