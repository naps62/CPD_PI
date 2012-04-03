#ifndef ___LIBTK___STOPWATCH_HPP___
#define ___LIBTK___STOPWATCH_HPP___

#include <iostream>
#include <sys/time.h>

using std::ostream;

/// General Tool Kit for a variety of useful stuff.
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

	/// Measures time intervals.
	/** This class is specially meant for execution time profilling.
	 * \todo Optimize this class with inline definitions.
	 */
	class Stopwatch
	{
		bool _running;    ///< Current state. Tracks whether the stopwatch has been started or is stopped.
		Time _control;    ///< Overhead control. Saves the value regarding the time interval spent starting/stopping the stopwatch.
		Time _begin;      ///< Timestamp tracking when the stopwatch was started.
		Time _end;        ///< Timestamp tracking when the stopwatch was stopped.
		Time _total;      ///< Total time the stopwatch was activated, since creation or last reset.
		Time _last;       ///< Duration of the last measured time interval.


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
		Time last();
	};
}

#endif//___LIBTK___STOPWATCH_HPP___
