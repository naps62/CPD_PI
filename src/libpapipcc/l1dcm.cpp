#include <papi/l1dcm.hpp>

namespace papi
{
	L1DataCacheMissesCounter::L1DataCacheMissesCounter()
	{
		this->add_event( _l1dcm.code() );
	}



	long long int
	L1DataCacheMissesCounter::misses()
	{
		return (*this)[ _l1dcm.code() ];
	}






	L1DataCacheMissesPresetEvent::L1DataCacheMissesPresetEvent()
	: Event( "PAPI_L1_DCM" )
	{}
}
