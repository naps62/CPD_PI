#ifndef ___PAPI_COUNTER_PRESET_LDINS___
#define ___PAPI_COUNTER_PRESET_LDINS___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct LoadInstructionsPresetEvent : public Event
	{
		LoadInstructionsPresetEvent();
	};





	class LoadInstructionsCounter : public PAPI_Custom
	{
		LoadInstructionsPresetEvent _ldins;
	public:
		LoadInstructionsCounter();

		//
		//  getter
		//
		long long int instructions();
	};
}

#endif//___PAPI_COUNTER_PRESET_LDINS___
