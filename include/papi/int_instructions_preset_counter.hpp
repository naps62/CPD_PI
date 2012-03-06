#ifndef ___INTEGER_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
#define ___INTEGER_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___

#include <papi/instructions_preset_counter.hpp>

namespace papi
{
	struct IntegerInstructionsPresetCounter
	: public InstructionsPresetCounter
	{
		//
		//    constructors
		//
		IntegerInstructionsPresetCounter();

		//
		//    getter
		//
		long long int instructions();
	};
}

#endif//___INTEGER_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
