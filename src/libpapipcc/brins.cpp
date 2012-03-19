#include <papi/brins.hpp>

namespace papi
{
	BranchInstructionsCounter::BranchInstructionsCounter()
	{
		this->add_event( _brins.code() );
	}



	long long int
	BranchInstructionsCounter::instructions()
	{
		return (*this)[ _brins.code() ];
	}






	BranchInstructionsPresetEvent::BranchInstructionsPresetEvent()
	: Event( "PAPI_BR_INS" )
	{}
}
