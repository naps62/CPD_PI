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

		struct L1InstructionCacheAccessesPresetEvent
		: public Event
		{
			L1InstructionCacheAccessesPresetEvent();
		};

		struct L1InstructionCacheMissesPresetEvent
		: public Event
		{
			L1InstructionCacheMissesPresetEvent();
		};

		struct L1TotalCacheAccessesPresetEvent
		: public Event
		{
			L1TotalCacheAccessesPresetEvent();
		};

		struct L1TotalCacheMissesPresetEvent
		: public Event
		{
			L1TotalCacheMissesPresetEvent();
		};





		struct L2TotalCacheAccessesPresetEvent
		: public Event
		{
			L2TotalCacheAccessesPresetEvent();
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



		struct L2InstructionCacheAccessesPresetEvent
		: public Event
		{
			L2InstructionCacheAccessesPresetEvent();
		};

		struct L2InstructionCacheMissesPresetEvent
		: public Event
		{
			L2InstructionCacheMissesPresetEvent();
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





		class L1CacheAccessesCounter
		: public CacheAccessesCounter
		{
			events::L1DataCacheAccessesPresetEvent        _l1dca;
			events::L1InstructionCacheAccessesPresetEvent _l1ica;
			events::L1TotalCacheAccessesPresetEvent       _l1tca;
		public:
			enum Event
			{
				DATA,
				INSTRUCTION,
				TOTAL
			};

			L1CacheAccessesCounter();

			long long int data();
			long long int instruction();
			long long int total();
			long long int accesses( Event e );
			long long int accesses();
		};



		/// Counter to measure the three types of misses that may occur in the L1 cache together.
		/**
		 * This is equivalent to using the data, instruction and total cache misses counters, but allows the measurement to be performed in a single run.
		 */
		class L1CacheMissesCounter
		: public CacheMissesCounter
		{
			events::L1DataCacheMissesPresetEvent        _l1dcm;
			events::L1InstructionCacheMissesPresetEvent _l1icm;
			events::L1TotalCacheMissesPresetEvent       _l1tcm;
		public:
			enum Event
			{
				DATA,
				INSTRUCTION,
				TOTAL
			};

			L1CacheMissesCounter();

			long long int data();
			long long int instruction();
			long long int total();
			long long int misses( Event e );
			long long int misses();
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



		class L1InstructionCacheMissesCounter
		: public CacheMissesCounter
		{
			events::L1InstructionCacheMissesPresetEvent _l1icm;
		public:
			L1InstructionCacheMissesCounter();

			long long int misses();
		};



		class L1TotalCacheMissesCounter
		: public CacheMissesCounter
		{
			events::L1TotalCacheMissesPresetEvent _l1tcm;
		public:
			L1TotalCacheMissesCounter();

			long long int misses();
		};





		class L2CacheAccessesCounter
		: public CacheAccessesCounter
		{
			events::L2DataCacheAccessesPresetEvent        _l2dca;
			events::L2InstructionCacheAccessesPresetEvent _l2ica;
			events::L2TotalCacheAccessesPresetEvent       _l2tca;
		public:
			enum Event
			{
				DATA,
				INSTRUCTION,
				TOTAL
			};

			L2CacheAccessesCounter();

			long long int data();
			long long int instruction();
			long long int total();
			long long int accesses( Event e );
			long long int accesses();
		};



		class L2CacheMissesCounter
		: public CacheMissesCounter
		{
			events::L2DataCacheMissesPresetEvent        _l2dcm;
			events::L2InstructionCacheMissesPresetEvent _l2icm;
			events::L2TotalCacheMissesPresetEvent       _l2tcm;
		public:
			enum Event
			{
				DATA,
				INSTRUCTION,
				TOTAL
			};

			L2CacheMissesCounter();

			long long int data();
			long long int instruction();
			long long int total();
			long long int misses( Event e );
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
		


		class L2InstructionCacheAccessesCounter
		: public CacheAccessesCounter
		{
			events::L2InstructionCacheAccessesPresetEvent _l2ica;
		public:
			L2InstructionCacheAccessesCounter();

			long long int accesses();
		};



		class L2InstructionCacheMissesCounter
		: public CacheMissesCounter
		{
			events::L2InstructionCacheMissesPresetEvent _l2icm;
		public:
			L2InstructionCacheMissesCounter();

			long long int misses();
		};



		class L2TotalCacheAccessesCounter
		: public CacheAccessesCounter
		{
			events::L2TotalCacheAccessesPresetEvent _l2tca;
		public:
			/// Default constructor. Adds the event \ref events::L2TotalCacheAccessesPresetEvent.
			L2TotalCacheAccessesCounter();

			//
			//  getter
			//

			/// Retrieves the last measured value for the number of accesses in the L2 cache (both for instructions and data).
			long long int accesses();
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
