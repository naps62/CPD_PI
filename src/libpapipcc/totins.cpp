#include <papi/totins.hpp>

namespace papi
{
	TotalInstructionsCounter::TotalInstructionsCounter()
	{
		this->add_event( _totins.code() );
	}



	long long int
	TotalInstructionsCounter::instructions()
	{
		return (*this)[ _totins.code() ];
	}






	TotalInstructionsPresetEvent::TotalInstructionsPresetEvent()
	: Event( "PAPI_TOT_INS" )
	{}
}
