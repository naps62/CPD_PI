#ifndef ___LOAD_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
#define ___LOAD_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___

#include <papi/instructions_preset_counter.hpp>

namespace papi
{
	struct LoadInstructionsPresetCounter
	: public InstructionsPresetCounter
	{
		//
		//    constructors
		//
		LoadInstructionsPresetCounter();

		//
		//    getter
		//
		long long int instructions();
	};
}

#endif//___LOAD_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
