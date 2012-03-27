#pragma once
#ifndef ___PAPI_STOPWATCH_HPP___
#define ___PAPI_STOPWATCH_HPP___

#include <map>
#include <vector>

#include <papi.h>

using std::map;
using std::vector;


namespace papi
{
	namespace time
	{
		namespace real
		{
			/// Measures time intervals between keypoints.
			/** Uses the PAPI_get_real_nsec() from the PAPI library as the time function.
			 */
			class Stopwatch
			{
				bool          _running;     ///< State variable: activated during the measurement.
				long long int _begin;       ///< Instant (in nanoseconds) when the last measurement began.
				long long int _end;         ///< Instant (in nanoseconds) when the last measurement ended.
				long long int _total;       ///< Time elapsed (in nanoseconds) for every measurement since creation or last reset.
				long long int _last;        ///< Time elapsed (in nanoseconds) for the last measurement.
				long long int _overhead;    ///< Overhead (in nanoseconds) of the measurement.
			public:
				Stopwatch();

				/// Starts measuring time.
				void start();

				/// Stops measuring time.
				void stop();

				/// Resets the elapsed time.
				void reset();

				/// Stop and reset stopwatch.
				/**
				 * \return Elapsed time (in nanoseconds) of the last measurement.
				 */
				long long int finish();
				
				/// Toggle the stopwatch state.
				void toggle();



				//
				//  getters
				//

				/// Retrieves the elapsed time, without stopping the measurement.
				/**
				 * \return Elapsed time (in nanoseconds), without a measured overhead.
				 * \warning The measurement is stopped before the time reading and started again after to reduce the impact of this function.
				 */
				long long int now() const;
				
				/// Retrieves the total elapsed time while this instance was activated.
				/**
				 * \return Elapsed time (in nanoseconds) while activated, since creation or last reset.
				 */
				long long int total() const;
				
				/// Retrieves the elapsed time during the last measurement.
				/**
				 * \return Elapsed time (in nanoseconds) for the last measurement.
				 */
				long long int last() const;
			};
		}
	}

}

#endif/*___PAPI_STOPWATCH_HPP___*/
