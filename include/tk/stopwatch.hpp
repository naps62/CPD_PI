#ifndef ___STOPWATCH_HPP___
#define ___STOPWATCH_HPP___

#include <iostream>
#include <sys/time.h>

using std::ostream;

namespace tk
{
	class Time : timespec
	{
		static timespec timespec_now();
		static timespec& timespec_now( timespec& time );


		//	instance
		time_t _seconds;
		long _nanoseconds;

		void set( const timespec& now );
		void set( time_t seconds, long nanoseconds );


		//	friends
		friend ostream& operator<<(ostream& out, const Time& time);


	public:
		Time();
		Time( const Time& time );

		void now();
		void reset();

		//	getters
		time_t seconds() const;
		long nanoseconds() const;

		//	operators
		Time& operator=( const Time& time );

		Time& operator+=( const Time& time );
		Time& operator-=( const Time& time );

		const Time operator+( const Time &time );
		const Time operator-( const Time &time );
	};

	class Stopwatch
	{
//		static Time now();
//		static Time & now(Time &time);
//		static Time add(Time const &time1, Time const &time2);
//		static Time diff(Time const &time1, Time const &time2);
//		static void reset(Time &time);


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

		//		worked
//		double total_s();
//		double total_ms();
//		double total_us();
//		unsigned long long int total_ns();
	};
}

#endif/*___STOPWATCH_HPP___*/
