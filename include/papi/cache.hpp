#ifndef ___PAPI_CACHE_HPP___
#define ___PAPI_CACHE_HPP___

#include <papi/event.hpp>
#include <papi/counter.hpp>

namespace papi
{
	namespace events
	{
		struct L1DataCacheAccessesPresetEvent
		: public Event
		{
			L1DataCacheAccessesPresetEvent();
		};

		struct L1DataCacheMissesPresetEvent
		: public Event
		{
			L1DataCacheMissesPresetEvent();
		};





		struct L2TotalCacheMissesPresetEvent
		: public Event
		{
			L2TotalCacheMissesPresetEvent();
		};



		struct L2DataCacheAccessesPresetEvent
		: public Event
		{
			L2DataCacheAccessesPresetEvent();
		};

		struct L2DataCacheMissesPresetEvent
		: public Event
		{
			L2DataCacheMissesPresetEvent();
		};
	}

	namespace counters
	{
		struct CacheAccessesCounter
		: public Counter
		{
			virtual long long int accesses() = 0;
		};

		struct CacheMissesCounter
		: public Counter
		{
			/// Retrieves the last measured value for the number of cache misses.
			virtual long long int misses() = 0;
		};



		class L1DataCacheAccessesCounter
		: public CacheAccessesCounter
		{
			events::L1DataCacheAccessesPresetEvent _l1dca;
		public:
			L1DataCacheAccessesCounter();

			long long int accesses();
		};

		class L1DataCacheMissesCounter
		: public CacheMissesCounter
		{
			events::L1DataCacheMissesPresetEvent _l1dcm;
		public:
			L1DataCacheMissesCounter();

			long long int misses();
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

		

		class L2DataCacheAccessesCounter
		: public CacheAccessesCounter
		{
			events::L2DataCacheAccessesPresetEvent _l2dca;
		public:
			L2DataCacheAccessesCounter();

			long long int accesses();
		};

		class L2DataCacheMissesCounter
		: public CacheMissesCounter
		{
			events::L2DataCacheMissesPresetEvent _l2dcm;
		public:
			/// Default constructor. Adds the event \ref events::L2DataCacheMissesPresetEvent.
			L2DataCacheMissesCounter();

			//
			//  getter
			//

			/// Retrieves the last measured value for the number of misses in the L2 data cache.
			long long int misses();
		};
	}
}

#endif//___PAPI_CACHE_HPP___
