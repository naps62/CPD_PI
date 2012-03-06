#include <papi/total_instructions_preset_counter.hpp>

#include <papi.h>

namespace papi
{
	TotalInstructionsPresetCounter::TotalInstructionsPresetCounter()
	{
		this->add_event( PAPI_TOT_INS );
	}

	long long int
	TotalInstructionsPresetCounter::instructions()
	{
		return (*this)[ PAPI_TOT_INS ];
	}
}
