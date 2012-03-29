#ifndef ___PAPI_CACHE_HPP___
#define ___PAPI_CACHE_HPP___

#include <papi/event.hpp>
#include <papi/counter.hpp>

namespace papi
{
	namespace events
	{
		struct L2TotalCacheMissesPresetEvent
		: public Event
		{
			L2TotalCacheMissesPresetEvent();
		};
	}

	namespace counters
	{
		struct CacheMissesCounter
		: public Counter
		{
			/// Retrieves the last measured value for the number of cache misses.
			virtual long long int misses() = 0;
		};

		/// Measures the total number of misses in the L2 cache using a preset counter.
		/** This counter includes both instructions and data misses.
		 *
		 * \sa events::L2TotalCacheMissesPresetEvent
		 */
		class L2TotalCacheMissesCounter
		: public CacheMissesCounter
		{
			events::L2TotalCacheMissesPresetEvent _l2tcm;
		public:
			/// Default constructor. Adds the event \ref events::L2TotalCacheMissesPresetEvent.
			L2TotalCacheMissesCounter();

			//
			//  getter
			//

			/// Retrieves the last measured value for the number of misses in the L2 cache (both for instructions and data).
			long long int misses();
		};
	}
}

#endif//___PAPI_CACHE_HPP___
