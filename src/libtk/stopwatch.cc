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
	timespec Time::timespec_now()
	{
		timespec now;
		timespec_now( now );
		return now;
	}

	timespec& Time::timespec_now( timespec& time )
	{
		if ( clock_gettime( CLOCK_REALTIME , &time ) )
			cerr
				<<	__FILE__
				<<	':'
				<<	__LINE__
				<<	"Error getting current timestamp."
				<<	endl;
		return time;
	}

	void Time::set( const timespec& time )
	{
		_seconds = time.tv_sec;
		_nanoseconds = time.tv_nsec;
	}

	void Time::set( time_t seconds , long nanoseconds )
	{
		_seconds = seconds;
		_nanoseconds = nanoseconds;
	}

	Time::Time()
	{
		this->now();
	}

	Time::Time( const Time& original ):
		_seconds( original.seconds() ),
		_nanoseconds( original.nanoseconds() )
	{}

	void Time::now()
	{
		this->set( Time::timespec_now() );
	}

	void Time::reset()
	{
		this->set( 0 , 0 );
	}

	//	getters
	time_t Time::seconds() const
	{
		return _seconds;
	}

	long Time::nanoseconds() const
	{
		return _nanoseconds;
	}

	Time& Time::operator=( const Time& time )
	{
		_seconds = time.seconds();
		_nanoseconds = time.nanoseconds();
		return *this;
	}

	Time& Time::operator+=( const Time& time )
	{
		_nanoseconds += time.nanoseconds();
		if ( _nanoseconds > NANOS_PER_SEC )
		{
			_seconds += time.seconds() + 1;
			_nanoseconds -= NANOS_PER_SEC;
		}
		else
		{
			_seconds += time.seconds();
		}
		return *this;
	}

	Time& Time::operator-=( const Time& time )
	{
		_nanoseconds -= time.nanoseconds();
		if ( _nanoseconds < 0 )
		{
			_seconds -= time.seconds() - 1;
			_nanoseconds += NANOS_PER_SEC;
		}
		else
		{
			_seconds -= time.seconds();
		}
		return *this;
	}

	const Time Time::operator+( const Time &time )
	{
		Time t = *this;
		t += time;
		return t;
	}

	const Time Time::operator-( const Time &time )
	{
		Time t = *this;
		t -= time;
		return t;
	}

	ostream& operator<<(ostream& out, const Time& time)
	{
		out
			<<	'['
			<<	time.seconds()
			<<	'+'
			<<	time.nanoseconds()
			<<	']';
		return out;
	}

	

	//
	//	Stopwatch class
	//
//	Time Stopwatch::add(Time const &time1, Time const &time2)
//	{
//		Time sum;
//		long dnsec = time1.tv_nsec + time2.tv_nsec;
//		if ( dnsec > NANOS_PER_SEC )
//		{
//			sum.tv_sec = time1.tv_sec + time2.tv_sec + 1;
//			sum.tv_nsec = dnsec - NANOS_PER_SEC;
//		}
//		else
//		{
//			sum.tv_sec = time1.tv_sec + time2.tv_sec;
//			sum.tv_nsec = dnsec;
//		}
//		{
//			cerr
//				<<	time1
//				<<	'+'
//				<<	time2
//				<<	'='
//				<<	sum
//				<<	endl;
//		}
//		return sum;
//	}

//	Time Stopwatch::diff(Time const &start, Time const &end)
//	{
//		Time diff;
//		long dnsec = end.tv_nsec - start.tv_nsec;
//		if ( dnsec < 0 )
//		{
//			diff.tv_sec = end.tv_sec - start.tv_sec - 1;
//			diff.tv_nsec = NANOS_PER_SEC + dnsec;
//		}
//		else
//		{
//			diff.tv_sec = end.tv_sec - start.tv_sec;
//			diff.tv_nsec = dnsec;
//		}
//		{
//			cerr
//				<<	end
//				<<	'-'
//				<<	start
//				<<	'='
//				<<	diff
//				<<	endl;
//		}
//		return diff;
//	}

//	void Stopwatch::reset(Time &time)
//	{
//		time.tv_sec = 0;
//		time.tv_nsec = 0;
//		{
//			cerr
//				<<	"Resetting time: "
//				<<	time
//				<<	endl;
//		}
//	}

	//	BEGIN INSTANCE
	Stopwatch::Stopwatch() : _running(false)
	{
//		Stopwatch::reset( _total );	
		_total.reset();
		_control.reset();
		this->start();
		this->stop();
		_control = _total;
		_total.reset();
	}

	void Stopwatch::start()
	{
		if ( ! _running )
		{
//			Stopwatch::reset( _partial );
			_partial.reset();
			_running = ( ! _running );
//			Stopwatch::now( _begin );
//			_begin = Stopwatch::now();
			_begin.now();
			{
				cerr
					<<	"Starting"
					<<	endl;
			}
		}
	}

	void Stopwatch::stop()
	{
//		Stopwatch::now( _end );
//		_end = Stopwatch::now();
		_end.now();
		if ( _running )
		{
			_running = ( ! _running );
//			_partial = Stopwatch::diff( _begin , _end );
			_partial = _end - _begin;
			_partial -= _control;
//			_total = Stopwatch::add( _total , _partial );
			_total += _partial;
			{
				cerr
					<<	"Stopping"
					<<	endl;
			}
		}
	}

	void Stopwatch::reset()
	{
//		Stopwatch::reset( _partial );
		_partial.reset();
		{
			cerr
				<<	"Resetting partial"
				<<	endl;
		}
		if ( ! _running )
		{
//			Stopwatch::reset( _total );
			_total.reset();
			{
				cerr
					<<	"Resetting total"
					<<	endl;
			}
		}
	}

	void Stopwatch::finish()
	{
		this->stop();
		this->reset();
		{
			cerr
				<<	"Finished"
				<<	endl;
		}
	}

	void Stopwatch::toggle()
	{
		if ( _running )
			this->stop();
		else
			this->start();
		{
			cerr
				<<	"Toggled"
				<<	endl;
		}
	}

	//	getters
	Time Stopwatch::total()
	{
		return _total;
	}

//	unsigned long long int Stopwatch::total_ns()
//	{
//		return _total.tv_sec * NANOS_PER_SEC + _total.tv_nsec;
//	}
//
//	double Stopwatch::total_s()
//	{
//		return this->total_ns() / 1e9;
//	}
//
//	double Stopwatch::total_ms()
//	{
//		return this->total_ns() / 1e6;
//	}
//
//	double Stopwatch::total_us()
//	{
//		return this->total_ns() / 1e3;
//	}

	Time Stopwatch::partial()
	{
		return _partial;
	}
}
