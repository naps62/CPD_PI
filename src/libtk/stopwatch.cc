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
		_seconds( original.get_seconds() ),
		_nanoseconds( original.get_nanoseconds() )
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
	time_t Time::get_seconds() const
	{
		return _seconds;
	}

	long Time::get_nanoseconds() const
	{
		return _nanoseconds;
	}

	long long int Time::nanoseconds() const
	{
		return _seconds * NANOS_PER_SEC + _nanoseconds;
	}

	double Time::microseconds() const
	{
		return _seconds * 1000000 + _nanoseconds * 1e-3;
	}

	double Time::miliseconds() const
	{
		return _seconds * 1000 + _nanoseconds * 1e-6;
	}

	double Time::seconds() const
	{
		return _seconds + _nanoseconds * 1e-9;
	}

	double Time::minutes() const
	{
		return this->seconds() / 60;
	}

	double Time::hours() const
	{
		return this->seconds() / 3600;
	}

	//	operators
	Time& Time::operator=( const Time& time )
	{
		_seconds = time.get_seconds();
		_nanoseconds = time.get_nanoseconds();
		return *this;
	}

	Time& Time::operator+=( const Time& time )
	{
		_nanoseconds += time.get_nanoseconds();
		if ( _nanoseconds > NANOS_PER_SEC )
		{
			_seconds += time.get_seconds() + 1;
			_nanoseconds -= NANOS_PER_SEC;
		}
		else
		{
			_seconds += time.get_seconds();
		}
		return *this;
	}

	Time& Time::operator-=( const Time& time )
	{
		_nanoseconds -= time.get_nanoseconds();
		if ( _nanoseconds < 0 )
		{
			_seconds -= time.get_seconds() - 1;
			_nanoseconds += NANOS_PER_SEC;
		}
		else
		{
			_seconds -= time.get_seconds();
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
			<<	time.get_seconds()
			<<	'+'
			<<	time.get_nanoseconds()
			<<	']';
		return out;
	}

	

	//
	//	Stopwatch class
	//
	Stopwatch::Stopwatch() : _running(false)
	{
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
			_partial.reset();
			_running = ( ! _running );
			_begin.now();
		}
	}

	void Stopwatch::stop()
	{
		_end.now();
		if ( _running )
		{
			_running = ( ! _running );
			_partial = _end - _begin;
			_partial -= _control;
			_total += _partial;
		}
	}

	void Stopwatch::reset()
	{
		_partial.reset();
		if ( ! _running )
			_total.reset();
	}

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

	//	getters
	Time Stopwatch::total()
	{
		return _total;
	}

	Time Stopwatch::partial()
	{
		return _partial;
	}
}
