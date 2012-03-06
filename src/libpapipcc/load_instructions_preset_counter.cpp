#include <papi/load_instructions_preset_counter.hpp>

#include <papi.h>

namespace papi
{
	LoadInstructionsPresetCounter::LoadInstructionsPresetCounter()
	{
		this->add_event( PAPI_LD_INS );
	}

	long long int
	LoadInstructionsPresetCounter::instructions()
	{
		return (*this)[ PAPI_LD_INS ];
	}
}
