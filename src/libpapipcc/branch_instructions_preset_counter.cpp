#include <papi/branch_instructions_preset_counter.hpp>

#include <papi.h>

namespace papi
{
	BranchInstructionsPresetCounter::BranchInstructionsPresetCounter()
	{
		this->add_event( PAPI_BR_INS );
	}

	long long int
	BranchInstructionsPresetCounter::instructions()
	{
		return (*this)[ PAPI_BR_INS ];
	}
}
