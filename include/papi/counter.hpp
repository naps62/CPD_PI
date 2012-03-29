#pragma once
#ifndef ___PAPI_COUNTER_HPP___
#define ___PAPI_COUNTER_HPP___

#include <papi/stopwatch.hpp>

#include <map>
#include <vector>

namespace papi
{
	/**
	 * \todo Add overhead control. This depends on the counter, as single value counters will need only a variable, and multiple values counters will require an array.
	 * \todo {
	 * Give some statistical meaning to the overhead control. Measure the overhead a few times and use the median. An alternative would be to calculate the overhead right before the beginning of the measurement.
	 * This would allow the measured overhead to suffer from most interferences the measurement will.
	 * }
	 */
	class Counter
	: protected time::real::Stopwatch
	{
		int                    _set;     ///< PAPI library event set identifier.
		long long int *        _values;  ///< Auxiliary array to retrieve the values from the PAPI library.
		map<int,long long int> _lasts;   ///< Mapping of each event identifier with the values retrieved in the last measurement.
		map<int,long long int> _totals;  ///< Mapping of each event identifier with the values retrieved in the last measurement.
		vector<int>            _events;  ///< Vector holding the added events.

	protected:
		/// Adds an event to this instance's event set.
		virtual
		void add_event (int event);
		
		/// Adds an array of events to this instance's event set.
		virtual
		void add_event (int *events_v, unsigned events_c);

	public:
		Counter ();

		/// Starts the measurement of the added events.
		void start ();

		///	Stops the measurement and stores the values measured for each added event.
		void stop ();

		/// Resets all values (time and performance counters).
		void reset ();



		//
		//  getters
		//

		/// Retrieves the total elapsed time while activated for this counter, since creation or last reset.
		/**
		 * \return Total elapsed time (in nanoseconds) while measuring, since creation or last reset.
		 */
		long long int total() const;

		/// Retrieves the elapsed time for the last measurement.
		/**
		 * \return Last measurement elapsed time (in nanoseconds).
		 */
		long long int last() const;

		/// Retrieves the last measured value for an event.
		/**
		 * \return Last measurement value returned by the PAPI library.
		 */
		long long int last (int event);

		/// Retrieves the total measured value for an event.
		/**
		 * \return Total sum of the values returned by the PAPI library in every measurement, since creation or last reset.
		 */
		long long int total (int event);

		/// Same as total(event).
		long long int operator[] (int event);
	};
}

#endif//___PAPI_COUNTER_HPP___
