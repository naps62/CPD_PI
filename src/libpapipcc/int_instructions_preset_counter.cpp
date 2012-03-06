#include <papi/int_instructions_preset_counter.hpp>

#include <papi.h>

namespace papi
{
	IntegerInstructionsPresetCounter::IntegerInstructionsPresetCounter()
	{
		this->add_event( PAPI_INT_INS );
	}

	long long int
	IntegerInstructionsPresetCounter::instructions()
	{
		return (*this)[ PAPI_INT_INS ];
	}
}
