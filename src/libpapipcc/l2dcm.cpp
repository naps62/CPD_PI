#include <papi/l2dcm.hpp>

namespace papi
{
	L2DataCacheMissesCounter::L2DataCacheMissesCounter()
	{
		this->add_event( _l2dcm.code() );
	}



	long long int
	L2DataCacheMissesCounter::misses()
	{
		return (*this)[ _l2dcm.code() ];
	}






	L2DataCacheMissesPresetEvent::L2DataCacheMissesPresetEvent()
	: Event( "PAPI_L2_DCM" )
	{}
}
