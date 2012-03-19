#ifndef ___PAPI_COUNTER_PRESET_SRINS___
#define ___PAPI_COUNTER_PRESET_SRINS___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct StoreInstructionsPresetEvent : public Event
	{
		StoreInstructionsPresetEvent();
	};





	class StoreInstructionsCounter : public PAPI_Custom
	{
		StoreInstructionsPresetEvent _srins;
	public:
		StoreInstructionsCounter();

		//
		//  getter
		//
		long long int instructions();
	};
}

#endif//___PAPI_COUNTER_PRESET_SRINS___
