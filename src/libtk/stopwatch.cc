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

	void Time::now()
	{
		this->set( Time::timernow() );
	}

	void Time::reset()
	{
		timerclear( this );
	}

	//	getters
	time_t Time::get_seconds() const
	{
		return this->tv_sec;
	}

	suseconds_t Time::get_microseconds() const
	{
		return this->tv_usec;
	}

	long long int Time::microseconds() const
	{
		return this->tv_sec * 1000000 + this->tv_usec;
	}

	double Time::miliseconds() const
	{
		return this->tv_sec * 1000 + this->tv_usec * 1e-3;
	}

	double Time::seconds() const
	{
		return this->tv_sec + this->tv_usec * 1e-6;
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
		this->set( time );
		return *this;
	}

	Time& Time::operator+=( const Time& time )
	{
		timeval s;
		timeradd(this,&time,&s);
		this->set(s);
		return *this;
	}

	Time& Time::operator-=( const Time& time )
	{
		timeval s;
		timersub(this,&time,&s);
		this->set(s);
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
			<<	time.get_microseconds()
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
