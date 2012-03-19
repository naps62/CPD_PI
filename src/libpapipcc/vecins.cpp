#include <papi/vecins.hpp>

namespace papi
{
	VectorInstructionsCounter::VectorInstructionsCounter()
	{
		this->add_event( _vecins.code() );
	}



	long long int
	VectorInstructionsCounter::instructions()
	{
		return (*this)[ _vecins.code() ];
	}






	VectorInstructionsPresetEvent::VectorInstructionsPresetEvent()
	: Event( "PAPI_VEC_INS" )
	{}
}
