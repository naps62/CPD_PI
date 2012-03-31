#include <papi/flops.hpp>

namespace papi
{
	namespace events
	{
		FloatingPointOperationsPresetEvent::FloatingPointOperationsPresetEvent()
		: Event( "PAPI_FP_OPS" )
		{}
	}

	namespace counters
	{
		FloatingPointOperationsCounter::FloatingPointOperationsCounter()
		{
			this->add_event( _flops.code() );
		}

		long long int
		FloatingPointOperationsCounter::operations()
		{
			return this->last( _flops.code() );
		}
	}
}
