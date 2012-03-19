#ifndef ___PAPI_COUNTER_PRESET_TOTINS___
#define ___PAPI_COUNTER_PRESET_TOTINS___

#include <papi/papi.hpp>
#include <papi/event.hpp>

namespace papi
{
	struct TotalInstructionsPresetEvent : public Event
	{
		TotalInstructionsPresetEvent();
	};





	class TotalInstructionsCounter : public PAPI_Custom
	{
		TotalInstructionsPresetEvent _totins;
	public:
		TotalInstructionsCounter();

		//
		//  getter
		//
		long long int instructions();
	};
}

#endif//___PAPI_COUNTER_PRESET_TOTINS___
