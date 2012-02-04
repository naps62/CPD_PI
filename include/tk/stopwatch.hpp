#ifndef ___STOPWATCH_HPP___
#define ___STOPWATCH_HPP___

#include <iostream>
#include <sys/time.h>

using std::ostream;

namespace tk
{
//	class Time : timespec
	class Time : public timeval
	{
//		static timespec timespec_now();
//		static timespec& timespec_now( timespec& time );
		static timeval timernow();
		static timeval& timernow( timeval& time );


		//	instance
//		time_t _seconds;
//		long _nanoseconds;
//		suseconds_t _microseconds;

//		void set( const timespec& now );
//		void set( time_t seconds, long nanoseconds );
		void set( const timeval& time );
		void set( const time_t seconds, const suseconds_t _microseconds );


		//	friends
		friend ostream& operator<<(ostream& out, const Time& time);


	public:
		Time();
		Time( const Time& time );

		void now();
		void reset();

		//	getters
		//		pure
		time_t get_seconds() const;
//		long get_nanoseconds() const;
		suseconds_t get_microseconds() const;
		//		worked
		double hours() const;
		double minutes() const;
		double seconds() const;
		double miliseconds() const;
		long long int microseconds() const;
//		double microseconds() const;
//		long long int nanoseconds() const;

		//	operators
		Time& operator=( const Time& time );

		Time& operator+=( const Time& time );
		Time& operator-=( const Time& time );

		const Time operator+( const Time &time );
		const Time operator-( const Time &time );
	};

	class Stopwatch
	{
		//	instance
		bool _running;
		Time _control;
		Time _begin;
		Time _end;
		Time _total;
		Time _partial;


		public:
		//	constructors
		Stopwatch();

		//	actions
		void start();
		void stop();
		void reset();
		void finish();
		void toggle();
		
		//	getters
		//		pure
		Time total();
		Time partial();
	};
}

#endif/*___STOPWATCH_HPP___*/
