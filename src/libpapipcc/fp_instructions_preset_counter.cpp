#include <papi/fp_instructions_preset_counter.hpp>

#include <papi.h>

namespace papi
{
	FloatingPointInstructionsPresetCounter::FloatingPointInstructionsPresetCounter()
	{
		this->add_event( PAPI_FP_INS );
	}

	long long int
	FloatingPointInstructionsPresetCounter::instructions()
	{
		return (*this)[ PAPI_FP_INS ];
	}
}
