#ifndef ___INSTRUCTIONS_PRESET_COUNTER_HPP___
#define ___INSTRUCTIONS_PRESET_COUNTER_HPP___

#include <papi/papi.hpp>

namespace papi
{
	struct InstructionsPresetCounter
	: public PAPI_Preset
	{
		//
		//    getter
		//
		virtual
		long long int instructions() = 0;
	};
}

#endif//___INSTRUCTIONS_PRESET_COUNTER_HPP___
