#include <papi/l2tcm.hpp>

namespace papi
{
	L2TotalCacheMissesCounter::L2TotalCacheMissesCounter()
	{
		this->add_event( _l2tcm.code() );
	}



	long long int
	L2TotalCacheMissesCounter::misses()
	{
		return (*this)[ _l2tcm.code() ];
	}






	L2TotalCacheMissesPresetEvent::L2TotalCacheMissesPresetEvent()
	: Event( "PAPI_L2_TCM" )
	{}
}
