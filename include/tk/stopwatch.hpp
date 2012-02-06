#ifndef ___STOPWATCH_HPP___
#define ___STOPWATCH_HPP___

#include <iostream>
#include <sys/time.h>

using std::ostream;

namespace tk
{
	class Time : public timeval
	{
		static timeval timernow();
		static timeval& timernow( timeval& time );


		//	instance
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
		suseconds_t get_microseconds() const;
		//		worked
		double hours() const;
		double minutes() const;
		double seconds() const;
		double miliseconds() const;
		long long int microseconds() const;

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
