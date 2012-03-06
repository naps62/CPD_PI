#include <papi/store_instructions_preset_counter.hpp>

#include <papi.h>

namespace papi
{
	StoreInstructionsPresetCounter::StoreInstructionsPresetCounter()
	{
		this->add_event( PAPI_SR_INS );
	}

	long long int
	StoreInstructionsPresetCounter::instructions()
	{
		return (*this)[ PAPI_SR_INS ];
	}
}
