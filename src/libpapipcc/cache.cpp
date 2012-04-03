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





		L2TotalCacheMissesPresetEvent::L2TotalCacheMissesPresetEvent()
		: Event( "PAPI_L2_TCM" )
		{}



		L2DataCacheAccessesPresetEvent::L2DataCacheAccessesPresetEvent()
		: Event( "PAPI_L2_DCA" )
		{}

		L2DataCacheMissesPresetEvent::L2DataCacheMissesPresetEvent()
		: Event( "PAPI_L2_DCM" )
		{}



		L2InstructionCacheMissesPresetEvent::L2InstructionCacheMissesPresetEvent()
		: Event( "PAPI_L2_ICM" )
		{}
	}

	namespace counters
	{
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



		L2TotalCacheMissesCounter::L2TotalCacheMissesCounter()
		{
			this->add_event( _l2tcm.code() );
		}

		long long int
		L2TotalCacheMissesCounter::misses()
		{
			return this->last( _l2tcm.code() );
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



		L2InstructionCacheMissesCounter::L2InstructionCacheMissesCounter()
		{
			this->add_event( _l2icm.code() );
		}

		long long int
		L2InstructionCacheMissesCounter::misses()
		{
			return this->last( _l2icm.code() );
		}
	}
}
