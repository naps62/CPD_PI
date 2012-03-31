#include <papi/cache.hpp>

namespace papi
{
	namespace events
	{
		L2TotalCacheMissesPresetEvent::L2TotalCacheMissesPresetEvent()
		: Event( "PAPI_L2_TCM" )
		{}

		L2DataCacheMissesPresetEvent::L2DataCacheMissesPresetEvent()
		: Event( "PAPI_L2_DCM" )
		{}
	}

	namespace counters
	{
		L2TotalCacheMissesCounter::L2TotalCacheMissesCounter()
		{
			this->add_event( _l2tcm.code() );
		}

		long long int
		L2TotalCacheMissesCounter::misses()
		{
			return this->last( _l2tcm.code() );
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
	}
}
