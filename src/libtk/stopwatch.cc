#include <tk/stopwatch.hpp>

using std::endl;
using std::cerr;

//
//	BEGIN CONSTANTS
//

#define NANOS_PER_SEC 1000000000

//
//	END CONSTANTS
//



//	BEGIN CLASS

namespace tk
{
	//
	//	Time class
	//
	timeval Time::timernow()
	{
		timeval now;
		return timernow( now );
	}

	timeval& Time::timernow( timeval& time )
	{
		gettimeofday( &time , NULL );
		return time;
	}

	void Time::set( const timeval& time )
	{
		this->tv_sec = time.tv_sec;
		this->tv_usec = time.tv_usec;
	}

	void Time::set( time_t seconds , suseconds_t microseconds )
	{
		this->tv_sec = seconds;
		this->tv_usec = microseconds;
	}

	Time::Time()
	{
		this->now();
	}

	Time::Time( const Time& original )
	{
		this->set( original );
	}

	double Time::minutes() const
	{
		return this->seconds() / 60;
	}

	double Time::hours() const
	{
		return this->seconds() / 3600;
	}

	

	//
	//	Stopwatch class
	//
	void Stopwatch::finish()
	{
		this->stop();
		this->reset();
	}

	void Stopwatch::toggle()
	{
		if ( _running )
			this->stop();
		else
			this->start();
	}
}
