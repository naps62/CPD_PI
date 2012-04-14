#include <papi/cache.hpp>

namespace papi
{
	namespace events
	{
		L1DataCacheAccessesPresetEvent::L1DataCacheAccessesPresetEvent()
		: Event( "PAPI_L1_DCA" )
		{}



		L1DataCacheMissesPresetEvent::L1DataCacheMissesPresetEvent()
		: Event( "PAPI_L1_DCM" )
		{}



		L1InstructionCacheAccessesPresetEvent::L1InstructionCacheAccessesPresetEvent()
		: Event( "PAPI_L1_ICA" )
		{}



		L1InstructionCacheMissesPresetEvent::L1InstructionCacheMissesPresetEvent()
		: Event( "PAPI_L1_ICM" )
		{}



		L1TotalCacheAccessesPresetEvent::L1TotalCacheAccessesPresetEvent()
		: Event( "PAPI_L1_TCA" )
		{}



		L1TotalCacheMissesPresetEvent::L1TotalCacheMissesPresetEvent()
		: Event( "PAPI_L1_TCM" )
		{}





		L2DataCacheAccessesPresetEvent::L2DataCacheAccessesPresetEvent()
		: Event( "PAPI_L2_DCA" )
		{}

		L2DataCacheMissesPresetEvent::L2DataCacheMissesPresetEvent()
		: Event( "PAPI_L2_DCM" )
		{}



		L2InstructionCacheAccessesPresetEvent::L2InstructionCacheAccessesPresetEvent()
		: Event( "PAPI_L2_ICA" )
		{}



		L2InstructionCacheMissesPresetEvent::L2InstructionCacheMissesPresetEvent()
		: Event( "PAPI_L2_ICM" )
		{}



		L2TotalCacheAccessesPresetEvent::L2TotalCacheAccessesPresetEvent()
		: Event( "PAPI_L2_TCA" )
		{}



		L2TotalCacheMissesPresetEvent::L2TotalCacheMissesPresetEvent()
		: Event( "PAPI_L2_TCM" )
		{}
	}

	namespace counters
	{
		L1CacheAccessesCounter::L1CacheAccessesCounter()
		{
			this->add_event( _l1dca.code() );
			this->add_event( _l1ica.code() );
			this->add_event( _l1tca.code() );
		}

		long long int
		L1CacheAccessesCounter::data()
		{
			return this->last( _l1dca.code() );
		}

		long long int
		L1CacheAccessesCounter::instruction()
		{
			return this->last( _l1ica.code() );
		}

		long long int
		L1CacheAccessesCounter::total()
		{
			return this->last( _l1tca.code() );
		}

		long long int
		L1CacheAccessesCounter::accesses( Event e )
		{
			switch (e)
			{
				case DATA:
					return this->data();
				case INSTRUCTION:
					return this->instruction();
				default:
					return this->total();
			}
		}

		long long int
		L1CacheAccessesCounter::accesses()
		{
			return this->accesses( TOTAL );
		}



		L1CacheMissesCounter::L1CacheMissesCounter()
		{
			this->add_event( _l1dcm.code() );
			this->add_event( _l1icm.code() );
			this->add_event( _l1tcm.code() );
		}

		long long int
		L1CacheMissesCounter::data()
		{
			return this->last( _l1dcm.code() );
		}

		long long int
		L1CacheMissesCounter::instruction()
		{
			return this->last( _l1icm.code() );
		}

		long long int
		L1CacheMissesCounter::total()
		{
			return this->last( _l1tcm.code() );
		}

		long long int
		L1CacheMissesCounter::misses( Event e )
		{
			switch (e)
			{
				case DATA:
					return this->data();
				case INSTRUCTION:
					return this->instruction();
				default:
					return this->total();
			}
		}

		long long int
		L1CacheMissesCounter::misses()
		{
			return this->misses( TOTAL );
		}



		L1DataCacheAccessesCounter::L1DataCacheAccessesCounter()
		{
			this->add_event( _l1dca.code() );
		}

		long long int
		L1DataCacheAccessesCounter::accesses()
		{
			return this->last( _l1dca.code() );
		}



		L1DataCacheMissesCounter::L1DataCacheMissesCounter()
		{
			this->add_event( _l1dcm.code() );
		}

		long long int
		L1DataCacheMissesCounter::misses()
		{
			return this->last( _l1dcm.code() );
		}



		L1InstructionCacheMissesCounter::L1InstructionCacheMissesCounter()
		{
			this->add_event( _l1icm.code() );
		}

		long long int
		L1InstructionCacheMissesCounter::misses()
		{
			return this->last( _l1icm.code() );
		}



		L1TotalCacheMissesCounter::L1TotalCacheMissesCounter()
		{
			this->add_event( _l1tcm.code() );
		}

		long long int
		L1TotalCacheMissesCounter::misses()
		{
			return this->last( _l1tcm.code() );
		}





		L2CacheAccessesCounter::L2CacheAccessesCounter()
		{
			this->add_event( _l2dca.code() );
			this->add_event( _l2ica.code() );
			this->add_event( _l2tca.code() );
		}

		long long int
		L2CacheAccessesCounter::data()
		{
			return this->last( _l2dca.code() );
		}

		long long int
		L2CacheAccessesCounter::instruction()
		{
			return this->last( _l2ica.code() );
		}

		long long int
		L2CacheAccessesCounter::total()
		{
			return this->last( _l2tca.code() );
		}

		long long int
		L2CacheAccessesCounter::accesses( Event e )
		{
			switch (e)
			{
				case DATA:
					return this->data();
				case INSTRUCTION:
					return this->instruction();
				default:
					return this->total();
			}
		}

		long long int
		L2CacheAccessesCounter::accesses()
		{
			return this->accesses( TOTAL );
		}



		L2CacheMissesCounter::L2CacheMissesCounter()
		{
			this->add_event( _l2dcm.code() );
			this->add_event( _l2icm.code() );
			this->add_event( _l2tcm.code() );
		}

		long long int
		L2CacheMissesCounter::data()
		{
			return this->last( _l2dcm.code() );
		}

		long long int
		L2CacheMissesCounter::instruction()
		{
			return this->last( _l2icm.code() );
		}

		long long int
		L2CacheMissesCounter::total()
		{
			return this->last( _l2tcm.code() );
		}

		long long int
		L2CacheMissesCounter::misses( Event e )
		{
			switch (e)
			{
				case DATA:
					return this->data();
				case INSTRUCTION:
					return this->instruction();
				default:
					return this->total();
			}
		}

		long long int
		L2CacheMissesCounter::misses()
		{
			return this->misses( TOTAL );
		}



		L2DataCacheAccessesCounter::L2DataCacheAccessesCounter()
		{
			this->add_event( _l2dca.code() );
		}

		long long int
		L2DataCacheAccessesCounter::accesses()
		{
			return this->last( _l2dca.code() );
		}



		L2DataCacheMissesCounter::L2DataCacheMissesCounter()
		{
			this->add_event( _l2dcm.code() );
		}

		long long int
		L2DataCacheMissesCounter::misses()
		{
			return this->last( _l2dcm.code() );
		}



		L2InstructionCacheAccessesCounter::L2InstructionCacheAccessesCounter()
		{
			this->add_event( _l2ica.code() );
		}

		long long int
		L2InstructionCacheAccessesCounter::accesses()
		{
			return this->last( _l2ica.code() );
		}



		L2InstructionCacheMissesCounter::L2InstructionCacheMissesCounter()
		{
			this->add_event( _l2icm.code() );
		}

		long long int
		L2InstructionCacheMissesCounter::misses()
		{
			return this->last( _l2icm.code() );
		}



		L2TotalCacheAccessesCounter::L2TotalCacheAccessesCounter()
		{
			this->add_event( _l2tca.code() );
		}

		long long int
		L2TotalCacheAccessesCounter::accesses()
		{
			return this->last( _l2tca.code() );
		}



		L2TotalCacheMissesCounter::L2TotalCacheMissesCounter()
		{
			this->add_event( _l2tcm.code() );
		}

		long long int
		L2TotalCacheMissesCounter::misses()
		{
			return this->last( _l2tcm.code() );
		}
	}
}
