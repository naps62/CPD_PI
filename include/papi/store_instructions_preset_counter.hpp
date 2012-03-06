#ifndef ___STORE_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
#define ___STORE_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___

#include <papi/instructions_preset_counter.hpp>

namespace papi
{
	struct StoreInstructionsPresetCounter
	: public InstructionsPresetCounter
	{
		//
		//    constructors
		//
		StoreInstructionsPresetCounter();

		//
		//    getter
		//
		long long int instructions();
	};
}

#endif//___STORE_INSTRUCTIONS_PRESET_COUNTER_PAPI_HPP___
