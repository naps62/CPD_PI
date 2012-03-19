#ifndef ___PAPI_COUNTER_PRESET_FPINS___
#define ___PAPI_COUNTER_PRESET_FPINS___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct FloatingPointInstructionsPresetEvent : public Event
	{
		FloatingPointInstructionsPresetEvent();
	};





	class FloatingPointInstructionsCounter : public PAPI_Custom
	{
		FloatingPointInstructionsPresetEvent _fpins;
	public:
		FloatingPointInstructionsCounter();

		//
		//  getter
		//
		long long int instructions();
	};
}

#endif//___PAPI_COUNTER_PRESET_FPINS___
