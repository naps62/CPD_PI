#ifndef ___TOTAL_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
#define ___TOTAL_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___

#include <papi/instructions_preset_counter.hpp>

namespace papi
{
	struct TotalInstructionsPresetCounter
	: public InstructionsPresetCounter
	{
		//
		//    constructors
		//
		TotalInstructionsPresetCounter();

		//
		//    getter
		//
		long long int instructions();
	};
}

#endif//___TOTAL_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
