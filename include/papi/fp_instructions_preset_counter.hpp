#ifndef ___FLOATINGPOINT_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
#define ___FLOATINGPOINT_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___

#include <papi/instructions_preset_counter.hpp>

namespace papi
{
	struct FloatingPointInstructionsPresetCounter
	: public InstructionsPresetCounter
	{
		//
		//    constructors
		//
		FloatingPointInstructionsPresetCounter();

		//
		//    getter
		//
		long long int instructions();
	};
}

#endif//___FLOATINGPOINT_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
