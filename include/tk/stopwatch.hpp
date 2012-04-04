#ifndef ___LIBTK___STOPWATCH_HPP___
#define ___LIBTK___STOPWATCH_HPP___

#include <iostream>
#include <sys/time.h>

using std::ostream;

/// General Tool Kit for a variety of useful stuff.
namespace tk
{
	/// Timestamp class, used with the Unix Standard library functions to retrieve the system time.
	/**
	 * This class has been optimized by implementing most methods as inline.
	 */
	class Time : public timeval
	{
		//
		//  STATIC
		//

		/// Retrieve the current timestamp.
		static timeval timernow();

		/// Retrieve the current timestamp.
		static timeval& timernow( timeval& time );


		
		//
		//  INSTANCE
		//
		
		void set( const timeval& time );
		void set( const time_t seconds, const suseconds_t _microseconds );


		//	friends
		friend ostream& operator<<(ostream& out, const Time& time);
		friend const Time operator-( const Time &t1 , const Time &t2 );


	public:
		/// Default constructor. Creates an instance which holds the current timestamp.
		Time();

		/// Copy constructor. Creates a new instance with the same timestamp as another.
		Time( const Time& time );

		/// Set this instance's timestamp to the current moment.
		void now();

		/// Clear this instance's timestamp.
		void reset();

		//
		//  GETTERS
		//

		/// Retrieve from the timeval structure only the value regarding the seconds fraction of the timestamp.
		time_t get_seconds() const;

		/// Retrieve from the timeval structure only the value regarding the microseconds fraction of the timestamp.
		suseconds_t get_microseconds() const;
		
		/// Retrieve this instance's timestamp value in hours.
		double hours() const;

		/// Retrieve this instance's timestamp value in minutes.
		double minutes() const;
		
		/// Retrieve this instance's timestamp value in seconds.
		double seconds() const;

		/// Retrieve this instance's timestamp value in miliseconds.
		double miliseconds() const;

		/// Retrieve this instance's timestamp value in microseconds.
		long long int microseconds() const;

		//
		//  OPERATORS
		//

		/// Assignment operator. Sets this instance's timestamp to the same value as another.
		Time& operator=( const Time& time );

		/// Modifying addition. Adds the timestamp of another instance to this instance's timestamp.
		Time& operator+=( const Time& time );

		/// Modifying difference. Changes this instance's timestamp to be the difference between the original timestamp and another instance's timestamp.
		Time& operator-=( const Time& time );

		const Time operator+( const Time &time );
		const Time operator-( const Time &time );

	};

	/// Measures time intervals.
	/** This class is specially meant for execution time profilling.
	 *
	 * This class has been optimized by implementing most methods as inline.
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
		/// Default constructor. Reset every timestamp and perform a test run to obtain the overhead control value.
		Stopwatch();

		/// Start counting time.
		/**
		 * Resets the partial timer, changes the current state and saves the current value of the time mechanism as the initial timestamp.
		 * This function is ignored if the stopwatch is running.
		 */
		void start();

		/// Stop counting time.
		/**
		 * Gets the current timestamp and toggles the state. The measured time interval is saved in the 'last' timer and added to the 'total'.
		 * This function is ignored if the stopwatch is stopped.
		 */
		void stop();

		/// Reset the timers.
		/**
		 * Changes both the 'last' and 'total' timers back to zero. The 'total' timer remains unchanged if the stopwatch is running.
		 */
		void reset();

		/// Stops and resets the stopwatch.
		void finish();

		/// Toggles the stopwatch state.
		/**
		 * If the stopwatch is running, stop is called. Otherwise start is called.
		 */
		void toggle();
		
		//
		//  GETTERS
		//

		/// Retrieve the 'total' timer value.
		/**
		 * The value of this timer is the total amount of time the stopwatch spent activated since creation or last call to the reset method.
		 */
		Time total();

		/// Retrieve the last measured timer value.
		Time last();
	};
}

#endif//___LIBTK___STOPWATCH_HPP___
