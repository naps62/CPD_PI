#include <papi/flops.hpp>

namespace papi
{
	FloatingPointOperationsCounter::FloatingPointOperationsCounter()
	{
		this->add_event( _flops.code() );
	}



	long long int
	FloatingPointOperationsCounter::operations()
	{
		return (*this)[ _flops.code() ];
	}






	FloatingPointOperationsPresetEvent::FloatingPointOperationsPresetEvent()
	: Event( "PAPI_FP_OPS" )
	{}
}
