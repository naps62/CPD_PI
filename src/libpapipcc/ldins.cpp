#include <papi/ldins.hpp>

namespace papi
{
	LoadInstructionsCounter::LoadInstructionsCounter()
	{
		this->add_event( _ldins.code() );
	}



	long long int
	LoadInstructionsCounter::instructions()
	{
		return (*this)[ _ldins.code() ];
	}






	LoadInstructionsPresetEvent::LoadInstructionsPresetEvent()
	: Event( "PAPI_LD_INS" )
	{}
}
