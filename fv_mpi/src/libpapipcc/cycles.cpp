#include <papi/cycles.hpp>

namespace papi
{
	namespace events
	{
		TotalCyclesPresetEvent::TotalCyclesPresetEvent()
		: Event( "PAPI_TOT_CYC" )
		{}
	}

	namespace counters
	{
		TotalCyclesCounter::TotalCyclesCounter()
		{
			this->add_event( _totcyc.code() );
		}

		long long int
		TotalCyclesCounter::cycles()
		{
			return this->last( _totcyc.code() );
		}
	}
}
