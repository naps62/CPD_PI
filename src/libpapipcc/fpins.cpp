#include <papi/fpins.hpp>

namespace papi
{
	FloatingPointInstructionsCounter::FloatingPointInstructionsCounter()
	{
		this->add_event( _fpins.code() );
	}



	long long int
	FloatingPointInstructionsCounter::instructions()
	{
		return (*this)[ _fpins.code() ];
	}






	FloatingPointInstructionsPresetEvent::FloatingPointInstructionsPresetEvent()
	: Event( "PAPI_FP_INS" )
	{}
}
