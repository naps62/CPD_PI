#include <papi/srins.hpp>

namespace papi
{
	StoreInstructionsCounter::StoreInstructionsCounter()
	{
		this->add_event( _srins.code() );
	}



	long long int
	StoreInstructionsCounter::instructions()
	{
		return (*this)[ _srins.code() ];
	}






	StoreInstructionsPresetEvent::StoreInstructionsPresetEvent()
	: Event( "PAPI_SR_INS" )
	{}
}
